#!/usr/bin/env python3
"""
Script to convert JSON polygon annotations to .npy mask files.
Reads polygon coordinates from JSON and creates binary masks.

Usage Examples:
    # Convert a single JSON file
    python convert_json_to_npy.py --input india_dataset/twoChannels_in/train/masks/223_2.json
    
    # Convert all second masks (matching corresponding images) and replace existing npy
    python convert_json_to_npy.py --input india_dataset/twoChannels_in/train/masks --auto-size
"""

import os
import json
import numpy as np
import cv2
import argparse
from pathlib import Path


def convert_json_to_npy(json_file, output_file=None, image_size=None):
    """
    Convert a JSON polygon annotation file to a numpy mask (.npy) file.
    
    Args:
        json_file: Path to the JSON file with polygon annotations
        output_file: Path to save the .npy file (default: same name as JSON with .npy extension)
        image_size: Tuple of (width, height). If None, tries to find corresponding image file
    
    Returns:
        The created mask as numpy array
    """
    # Load JSON annotations
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    
    # Determine image size
    if image_size is None:
        # Try to find corresponding image file
        json_path = Path(json_file)
        # Get the image directory (assuming masks are in ../images or parent/images)
        parent_dir = json_path.parent.parent
        image_dir = parent_dir / 'images'
        
        # Construct image filename
        image_filename = json_path.stem + '.png'
        image_path = image_dir / image_filename
        
        if image_path.exists():
            from PIL import Image
            img = Image.open(image_path)
            image_size = img.size  # (width, height)
            print(f"Found corresponding image: {image_path} with size {image_size}")
        else:
            # Default size
            image_size = (512, 512)
            print(f"Warning: Image not found at {image_path}, using default size {image_size}")
    else:
        print(f"Using provided size: {image_size}")
    
    width, height = image_size
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw each polygon on the mask
    for annotation in annotations:
        if 'content' in annotation and annotation.get('contentType') == 'polygon':
            # Extract polygon points
            points = annotation['content']
            polygon = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.int32)
            
            # Fill the polygon with white (255)
            cv2.fillPoly(mask, [polygon], 255)
    
    # Convert to binary (0 or 1) if needed
    # Scale down from 255 to 1
    mask_binary = (mask > 0).astype(np.uint8)
    
    print(f"Created mask shape: {mask_binary.shape}")
    print(f"Mask dtype: {mask_binary.dtype}")
    print(f"Min: {mask_binary.min()}, Max: {mask_binary.max()}")
    print(f"Non-zero pixels: {np.count_nonzero(mask_binary)} / {mask_binary.size} ({100*np.count_nonzero(mask_binary)/mask_binary.size:.1f}%)")
    
    # Determine output path
    if output_file is None:
        output_file = str(Path(json_file).with_suffix('.npy'))
    
    # Save as .npy file
    np.save(output_file, mask_binary)
    print(f"Saved mask to: {output_file}\n")
    
    return mask_binary


def convert_directory(mask_dir, auto_size=True):
    """
    Convert all JSON files in a directory to .npy files.
    
    Args:
        mask_dir: Directory containing JSON files
        auto_size: If True, automatically find image sizes from corresponding images
    """
    mask_dir = Path(mask_dir)
    
    # Find all JSON files
    json_files = list(mask_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files in {mask_dir}")
    
    for json_file in sorted(json_files):
        try:
            print(f"\nProcessing: {json_file}")
            convert_json_to_npy(json_file, image_size=None if auto_size else (512, 512))
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Convert JSON polygon annotations to .npy mask files')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input JSON file or directory containing JSON files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .npy file (only for single file conversion)')
    parser.add_argument('--size', type=str, default=None,
                        help='Image size as width,height (e.g., 512,512). If not provided, tries to find from corresponding image')
    parser.add_argument('--auto-size', action='store_true',
                        help='Automatically find image sizes from corresponding images in the images directory')
    
    args = parser.parse_args()
    
    # Parse size if provided
    image_size = None
    if args.size:
        try:
            width, height = map(int, args.size.split(','))
            image_size = (width, height)
        except ValueError:
            print(f"Error: Invalid size format '{args.size}'. Use width,height (e.g., 512,512)")
            return
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        convert_json_to_npy(args.input, args.output, image_size)
    elif input_path.is_dir():
        # Directory
        convert_directory(args.input, auto_size=args.auto_size)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return


if __name__ == '__main__':
    main()

