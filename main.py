import numpy as np
import av
import cv2
import logging
from pathlib import Path
from tqdm import tqdm


DATA_PATH = Path("/home/deck/Datasets/project-NIR-Camera")
MAIN_PATH = Path("/home/deck/Programs/NIR-Camera")
# Default to .mkv for FFV1 mathematically lossless encoding
OUT_PATH = MAIN_PATH / "output_lossless.mkv"

logger = logging.getLogger(__name__)
logging.basicConfig(filename="nir-camera.log", level=logging.DEBUG, filemode="w")

SCALE_FACTOR = 0.25

def siftAlign(img_ref, img_match, prev_transform):
    """
    Apply the sift transform. Tutorial: https://docs.opencv.org/3.4.8/dc/dc3/tutorial_py_matcher.html
    All adjacent frames must be similarly transformed, therefore we can pre-transform each frame
    with the previously calculated affine transform, hopefully speeding up the brute force matcher.

    To account for this shift, the new transform needs to be updated relative to the previous transform.
    Shift can be summed. Rotation needs to be multiplied e^i*2pi*theta_0 * e^i*2pi*theta_1 = e^i*2pi*(theta_0 + theta_1)
    """

    # Apply transform to provide easier matching
    if prev_transform is not None:
        img_match = applyTransform(img_match, prev_transform)

    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(img_ref, None)
    kp_mat, des_mat = sift.detectAndCompute(img_match, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_ref, des_mat, k=2)

    # Apply Lowe's ratio test to filter weak matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract coordinates
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_mat[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # estimateAffinePartial2D restricts the transform from skew transforms
    transform, mask = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.99
    )

    # Fallback to previous if failed to calculate new transform
    if transform is None:
        transform = prev_transform
    elif prev_transform is not None:
        # update the transform with the previous applied transform
        # T = A + B, A is rotation transform, B is the shift transform.
        # To update A (T[:, :2]), the new angle needs to be calculated.
        # To update B (T[:, 2]), it is simply the sum
        M = np.array(transform[:, 0]) + 1j * np.array(transform[:, 1])
        p_M = np.array(prev_transform[:, 0]) + 1j * np.array(prev_transform[:, 1])
        M = M * p_M
        transform[:, :2] = [[M[0].real, M[0].imag], [M[1].imag, -M[1].real]]
        transform[:, 2] = transform[:, 2] + prev_transform[:, 2]

    return transform

def applyTransform(img_align, xy_transform):
    
    aligned = np.empty_like(img_align, dtype=np.uint8)
    w, h = aligned.shape
    
    aligned = cv2.warpAffine(
        src = img_align, 
        M = xy_transform,   # Input with identity to avoid scaling and shearing
        dsize = (h, w),       # Same size as the input image
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return aligned

def applyMask(img, affine_tranforms):
    """
    Masks out the non-overlapping regions resulting from transforms.
    Honestly, just looks worse than keeping the mono-colour areas.
    """

    w, h, _ = img.shape
    mask = np.ones((w, h, 2), dtype=np.uint8)
    
    for idx in range(2):
        mask[:, :, idx] = cv2.warpAffine(
            src = mask[:, :, idx], 
            M = affine_tranforms[idx],   # Input with identity to avoid scaling and shearing
            dsize = (h, w),       # Same size as the input image
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    mask = np.bitwise_and(mask[:, :, 0], mask[:, :, 1])
    img = img * mask[:, :, np.newaxis]

    return img  


def combine_multispectral_to_rgb(data_path, output_path, file_prefix, masked=False, fix_contrast=False, lossless=True):
    # Open Input Containers and Streams
    wavelengths = (750, 850, 940)
    levels = (0.7, 0.7, 1.0)
    containers = {}
    streams = {}
    
    for wl in wavelengths:
        containers[wl] = av.open(str(data_path / f"{file_prefix}_{wl}nm.mp4"), "r")
        streams[wl] = containers[wl].streams.video[0]

    # Configure the Output Container
    out_container = av.open(str(output_path), mode='w')

    ref_stream = streams[750]
    fps = ref_stream.average_rate

    if lossless:
        # FFV1 is PyAV's native mathematically lossless video codec
        out_stream = out_container.add_stream('ffv1', rate=fps)
        out_stream.pix_fmt = 'bgr0'  # FFV1 does not support rgb24
        print("Encoding mode: FFV1 (Lossless)")
    else:
        # Fallback to lossy H.264/OpenH264/MPEG4 for lightweight preview files
        try:
            out_stream = out_container.add_stream('h264', rate=fps)
        except Exception:
            out_stream = out_container.add_stream('mpeg4', rate=fps)
        out_stream.pix_fmt = 'yuv420p'
        print("Encoding mode: H.264/MPEG4 (Lossy)")
    
    out_stream.width = ref_stream.codec_context.width
    out_stream.height = ref_stream.codec_context.height
    
    print(f"Processing at {fps} FPS. Output dimensions: {out_stream.width}x{out_stream.height}")

    # Synchronous Frame Processing Loop

    decoders = [containers[wl].decode(streams[wl]) for wl in wavelengths]
    total_frames = ref_stream.frames if ref_stream.frames > 0 else None

    img = np.ndarray((out_stream.height, out_stream.width, 3), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Configure scaling factor for alignment speedup (e.g., 0.25 is 1/4th resolution)
    affine_transform_750 = None
    affine_transform_940 = None

    for idx, frames in enumerate(tqdm(zip(*decoders), total=total_frames, desc="Aligning Frames", unit="frames")):
        
        # Convert to numpy
        img[:, :, 0] = frames[0].to_ndarray(format='gray')
        img[:, :, 1] = frames[1].to_ndarray(format='gray')
        img[:, :, 2] = frames[2].to_ndarray(format='gray')

        # # Downsample
        small_img = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)
        x, y, _ = small_img.shape

        # # Find affine transform on downsampled images
        affine_transform_750 = siftAlign(small_img[:, :, 1], small_img[:, :, 0], affine_transform_750)
        affine_transform_940 = siftAlign(small_img[:, :, 1], small_img[:, :, 2], affine_transform_940)

        # # Upscale translational transform
        upscaled_750 = affine_transform_750
        upscaled_940 = affine_transform_940
        upscaled_750[:, 2] = upscaled_750[:, 2] / SCALE_FACTOR
        upscaled_940[:, 2] = upscaled_940[:, 2] / SCALE_FACTOR

        img[:, :, 0] = applyTransform(img[:, :, 0], upscaled_750)
        img[:, :, 2] = applyTransform(img[:, :, 2], upscaled_940)

        # Optional to remove non-overlapping image regions
        if masked:
            img = applyMask(img, [upscaled_750, upscaled_940])

        # Optional to apply CLAHE: https://docs.opencv.org/4.1.1/d5/daf/tutorial_py_histogram_equalization.html
        if fix_contrast:
            img[:, :, 0] = clahe.apply(img[:, :, 0])
            img[:, :, 1] = clahe.apply(img[:, :, 1])
            img[:, :, 2] = clahe.apply(img[:, :, 2])

            # Numpy arrays override python multiplier method __mul__
            img[:, :, 0] = img[:, :, 0] * levels[0]
            img[:, :, 1] = img[:, :, 0] * levels[1]
            img[:, :, 2] = img[:, :, 0] * levels[2]

        # --- PYAV HANDOFF ---
        new_frame = av.VideoFrame.from_ndarray(img, format='rgb24')

        # Encode and Mux
        for packet in out_stream.encode(new_frame):
            out_container.mux(packet)
            
    # Flush the Encoder
    for packet in out_stream.encode():
        out_container.mux(packet)
        
    # Clean up
    for wl in wavelengths:
        containers[wl].close()
    out_container.close()
    print("Processing complete.")

if __name__ == "__main__":
    # Set lossless=True to use FFV1, or lossless=False to write lossy .mp4 previews

    # Assumes format {file_name}_{wavelength}nm.mp4
    file_name = "scene1"
    combine_multispectral_to_rgb(DATA_PATH, OUT_PATH, file_prefix=file_name, masked=True, fix_contrast=False, lossless=True)