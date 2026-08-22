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
logging.basicConfig(filename="nir-camera-mistiming.log", level=logging.DEBUG, filemode="w")

SCALE_FACTOR = 0.25

def siftAlign(img_ref, img_match, prev_transform):
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

    # estimateAffinePartial2D restricts the transform
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
        # T = A + B, A is rotation transform, B is the translational transform.
        # To update A (T[:, :2]), the new angle needs to be calculated.
        # To update B (T[:, 2]), it is simply the sum
        M = np.array(transform[:, 0]) + 1j * np.array(transform[:, 1])
        p_M = np.array(prev_transform[:, 0]) + 1j * np.array(prev_transform[:, 1])
        M = M * p_M
        transform[:, :2] = [[M[0].real, M[0].imag], [M[1].imag, -M[1].real]]
        transform[:, 2] = transform[:, 2] + prev_transform[:, 2]

    return transform

def interpolateFrame(frame1, frame2, t=0.5):
    """
    Synthesizes a frame at time 't' (0.0 to 1.0) between frame1 and frame2.
    """
    
    # 1. Calculate dense optical flow vectors (u, v) for every pixel
    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2, None, 
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=9, poly_sigma=1.7, flags=0
    )
    
    # 2. Generate a coordinate grid for the image
    h, w = frame1.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 3. Scale the flow vectors by the time offset 't'
    # We subtract the flow to perform a backward warp, preventing "holes" in the output
    map_x_warp = np.float32(map_x - flow[..., 0] * t)
    map_y_warp = np.float32(map_y - flow[..., 1] * t)
    
    # 4. Remap the pixels to their new sub-frame coordinates
    interpolated_frame = cv2.remap(
        frame1, 
        map_x_warp, map_y_warp, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return interpolated_frame

def findTimingOffset(frame_ref, frame_sample):
    """
    Find timing offset over a series of frames from one camera to another.

    1. Find frame offsets
    2. Calculate phase correlation
    3. Try small interpolation
    4. Recalculate phase correlation
    5. Update interpolation value

    """
    return cv2.phaseCorrelate(frame_ref, frame_sample)
    


def applyTransform(img_align, affine_transform):
    
    aligned = np.empty_like(img_align, dtype=np.uint8)
    w, h = aligned.shape
    
    aligned = cv2.warpAffine(
        src = img_align, 
        M = affine_transform,   # Input with identity to avoid scaling and shearing
        dsize = (h, w),       # Same size as the input image
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return aligned

def mistimingFix(data_path, output_path, lossless=True):
    # 1. Open Input Containers and Streams
    wavelengths = (750, 850, 940)
    containers = {}
    streams = {}
    
    for wl in wavelengths:
        containers[wl] = av.open(str(data_path / f"scene4_{wl}nm.mp4"), "r")
        # Gets ffmpeg reference to a video stream in the container
        streams[wl] = containers[wl].streams.video[0]

    # 2. Configure the Output Container
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

    # 3. Collect decoder generators

    decoders = [containers[wl].decode(streams[wl]) for wl in wavelengths]
    total_frames = ref_stream.frames if ref_stream.frames > 0 else None

    img = np.ndarray((out_stream.height, out_stream.width, 3), dtype=np.uint8)
    prev_img = np.empty_like(img)

    affine_transform_750 = None
    affine_transform_940 = None

    # 4. Iterate through generators offset from one another to find timing offset.
    # Find approximate frame offset dx and dy using phaseCorrelate
    # If dx or dy are above a certain value, move onto finding finer timing offset
    prev_img[:, :, 0] = next(decoders[0]).to_ndarray(format='gray')
    prev_img[:, :, 1] = next(decoders[1]).to_ndarray(format='gray')
    prev_img[:, :, 2] = next(decoders[2]).to_ndarray(format='gray')
    prev_img = cv2.resize(prev_img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)
    affine_transform_750 = siftAlign(prev_img[:, :, 1], prev_img[:, :, 0], affine_transform_750)
    affine_transform_940 = siftAlign(prev_img[:, :, 1], prev_img[:, :, 2], affine_transform_940)
    prev_img[:, :, 0] = applyTransform(prev_img[:, :, 0], affine_transform_750)
    prev_img[:, :, 2] = applyTransform(prev_img[:, :, 2], affine_transform_940)  

    # Need to first ruggedly align frames to get accuracte phase correlation
    for idx, frames in enumerate(tqdm(zip(*decoders), total=total_frames-1, desc="Aligning Frames", unit="frames")):
        
        # Convert to numpy
        img[:, :, 0] = frames[0].to_ndarray(format='gray')
        img[:, :, 1] = frames[1].to_ndarray(format='gray')
        img[:, :, 2] = frames[2].to_ndarray(format='gray')

        # Downsample
        small_img = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)

        # Find affine transform on downsampled images
        affine_transform_750 = siftAlign(small_img[:, :, 1], small_img[:, :, 0], affine_transform_750)
        affine_transform_940 = siftAlign(small_img[:, :, 1], small_img[:, :, 2], affine_transform_940)

        # Adjust by synthesising new frame
        small_img[:, :, 0] = applyTransform(small_img[:, :, 0], affine_transform_750)
        small_img[:, :, 2] = applyTransform(small_img[:, :, 2], affine_transform_940)  

        # Synthesise new frame
        small_img[:, :, 0] = interpolateFrame(small_img[:, :, 0], prev_img[:, :, 0], t=0.30)

        # Using 850nm as reference
        # Full resolution transform
        # --- PYAV HANDOFF ---
        new_frame = av.VideoFrame.from_ndarray(small_img, format='rgb24')

        prev_img = small_img.copy()

        # Encode and Mux
        for packet in out_stream.encode(new_frame):
            out_container.mux(packet)
            
    # 6. Flush the Encoder
    for packet in out_stream.encode():
        out_container.mux(packet)

      

       
    # Clean up
    for wl in wavelengths:
        containers[wl].close()
    out_container.close()
    print("Processing complete.")

if __name__ == "__main__":
    # Set lossless=True to use FFV1, or lossless=False to write lossy .mp4 previews
    mistimingFix(DATA_PATH, OUT_PATH, lossless=True)