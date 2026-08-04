import os
import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from framework import RAWRenderer

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("render-utils")

def resize_short_side(image: np.ndarray, short_side_len: int = 500) -> np.ndarray:
    """
    Resizes an image so that its shortest side is equal to short_side_len,
    maintaining the aspect ratio.
    """
    h, w = image.shape[:2]
    if h < w:
        # Height is the short side
        scale = short_side_len / h
        new_h = short_side_len
        new_w = int(w * scale)
    else:
        # Width is the short side
        scale = short_side_len / w
        new_w = short_side_len
        new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def process_images(input_dir: str, output_dir: str, short_side: int = 500):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.is_dir():
        logger.error(f"Input directory {input_dir} does not exist.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Common RAW extensions
    raw_extensions = {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.orf'}
    raw_files = [f for f in input_path.iterdir() if f.suffix.lower() in raw_extensions]

    if not raw_files:
        logger.info("No RAW images found in the input directory.")
        return

    logger.info(f"Found {len(raw_files)} images. Rendering to {output_dir}...")

    for raw_file in tqdm(raw_files, desc="Rendering Images"):
        try:
            # Look for corresponding XMP file
            xmp_file = raw_file.with_suffix(".xmp")
            if not xmp_file.exists():
                # Try uppercase .XMP
                xmp_file_upper = raw_file.with_suffix(".XMP")
                xmp_file = xmp_file_upper if xmp_file_upper.exists() else None
            
            # Render image using RAWRenderer from framework.py
            # RAWRenderer.render returns RGB numpy array
            img_rgb = RAWRenderer.render(raw_file, xmp_file)
            
            # Resize image
            resized_img = resize_short_side(img_rgb, short_side)
            
            # Convert RGB to BGR for OpenCV saving
            img_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
            
            # Save as JPG
            output_file = output_path / (raw_file.stem + ".jpg")
            cv2.imwrite(str(output_file), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
        except Exception as e:
            logger.error(f"Failed to process {raw_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render RAW images to JPGs with a specific short-side width.")
    parser.add_argument("input_dir", help="Directory containing raw images")
    parser.add_argument("output_dir", help="Directory to save rendered JPGs")
    parser.add_argument("--short_side", type=int, default=500, help="Short side pixel width (default: 500)")

    args = parser.parse_args()
    process_images(args.input_dir, args.output_dir, args.short_side)