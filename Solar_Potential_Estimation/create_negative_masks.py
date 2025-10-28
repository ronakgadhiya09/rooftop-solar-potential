#!/usr/bin/env python3
"""
Create completely black (all zeros) masks for images that don't have tilted roofs.
These will be used as negative examples for training.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image


def create_black_mask(image_path, output_path, image_size=None):
    """
    Create a completely black mask (all zeros) for an image.
    
    Args:
        image_path: Path to the image file
        output_path: Path to save the black mask .npy file
        image_size: Tuple (width, height). If None, reads from image
    """
    # Get image size
    if image_size is None:
        img = Image.open(image_path)
        image_size = img.size  # (width, height)
    
    width, height = image_size
    
    # Create completely black mask (all zeros)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Save as .npy file
    np.save(output_path, mask)
    
    print(f"Created black mask: {Path(output_path).name} (Size: {width}x{height})")


def create_negative_masks_for_training(num_negative=20):
    """
    Create black masks for images without tilted roofs (negative examples).
    
    Args:
        num_negative: Number of negative examples to create (default: 20)
    """
    base_dir = Path("india_dataset/twoChannels_in/train")
    images_dir = base_dir / "images"
    json_dir = base_dir / "json_masks"
    output_dir = base_dir / "tilted_masks"
    
    # Get all image files
    image_files = set(f.stem for f in images_dir.glob("*.png"))
    
    # Get all json files (which have tilted masks)
    json_files = set(f.stem for f in json_dir.glob("*.json"))
    
    # Find images without tilted masks
    images_without_tilted = sorted(list(image_files - json_files))
    
    print(f"Total images: {len(image_files)}")
    print(f"Images with tilted masks: {len(json_files)}")
    print(f"Images without tilted masks: {len(images_without_tilted)}")
    print(f"\nCreating {num_negative} negative examples (black masks)...\n")
    
    # Select first num_negative images
    selected_images = images_without_tilted[:num_negative]
    
    created_count = 0
    skipped_count = 0
    
    for img_name in selected_images:
        image_path = images_dir / f"{img_name}.png"
        output_path = output_dir / f"{img_name}.npy"
        
        # Skip if mask already exists
        if output_path.exists():
            print(f"Skipping {img_name}.npy (already exists)")
            skipped_count += 1
            continue
        
        try:
            create_black_mask(image_path, output_path)
            created_count += 1
        except Exception as e:
            print(f"Error creating mask for {img_name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Created: {created_count} new black masks")
    print(f"  Skipped: {skipped_count} (already exist)")
    print(f"  Output directory: {output_dir}")
    print(f"\nNegative examples created:")
    for img_name in selected_images[:created_count]:
        print(f"  - {img_name}.npy")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create black masks for negative examples (non-tilted roofs)')
    parser.add_argument('--num', '-n', type=int, default=20,
                        help='Number of negative examples to create (default: 20)')
    
    args = parser.parse_args()
    
    create_negative_masks_for_training(num_negative=args.num)

