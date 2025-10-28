#!/usr/bin/env python3
"""
Script to convert .npy files to PNG images.
Supports binary masks, multi-channel masks, and labeled masks.

Usage Examples:
    # Convert a single .npy file
    python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks/1_1.npy
    
    # Convert all .npy files in a directory
    python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks
    
    # Convert all .npy files with specific output directory
    python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks --output converted_masks
    
    # Convert .npy files recursively in subdirectories
    python convert_npy_to_images.py --input india_dataset --recursive
    
    # Convert without normalization (useful for labeled masks with specific values)
    python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks --no-normalize
"""

import os
import numpy as np
from PIL import Image
import argparse


def convert_npy_to_image(npy_file, output_dir=None, normalize=True):
    """
    Convert a single .npy file to a PNG image.
    
    Args:
        npy_file: Path to the .npy file
        output_dir: Directory to save the output image (default: same as npy file)
        normalize: If True, normalize values to 0-255 range for visualization
    """
    # Load numpy array
    arr = np.load(npy_file)
    
    print(f"Processing: {npy_file}")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min: {arr.min()}, Max: {arr.max()}")
    
    # Handle different dimensions
    if len(arr.shape) == 3:
        # If 3D array with depth 1, squeeze it
        if arr.shape[2] == 1:
            arr = arr.squeeze(2)
        else:
            # Multi-channel: convert to grayscale or RGB
            if arr.shape[2] == 3:
                # Already RGB-like
                arr = arr
            else:
                # Take first channel or convert to grayscale
                arr = arr[:, :, 0]
    
    # Normalize to 0-255 range if needed
    if normalize:
        # Check if it's binary (0s and 1s) regardless of dtype
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            # Normalize from current range to 0-255
            arr = arr.astype(np.float32)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(arr.shape) == 2:
        # Grayscale
        img = Image.fromarray(arr, mode='L')
    else:
        # RGB or multi-channel
        img = Image.fromarray(arr)
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(npy_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(npy_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_converted.png")
    
    # Save image
    img.save(output_path)
    print(f"  Saved to: {output_path}\n")
    
    return output_path


def convert_directory(npy_dir, output_dir=None, recursive=False):
    """
    Convert all .npy files in a directory to PNG images.
    
    Args:
        npy_dir: Directory containing .npy files
        output_dir: Directory to save output images (default: npy_dir/images)
        recursive: If True, search subdirectories recursively
    """
    npy_dir = os.path.abspath(npy_dir)
    
    # Find all .npy files
    npy_files = []
    if recursive:
        for root, dirs, files in os.walk(npy_dir):
            for file in files:
                if file.endswith('.npy'):
                    npy_files.append(os.path.join(root, file))
    else:
        npy_files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')]
    
    print(f"Found {len(npy_files)} .npy files")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(npy_dir, "images")
    
    # Convert each file
    for npy_file in sorted(npy_files):
        try:
            convert_npy_to_image(npy_file, output_dir)
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert .npy files to PNG images')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input .npy file or directory containing .npy files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: same as input for single file, input/images for directory)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search for .npy files recursively in subdirectories')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Do not normalize values to 0-255 range')
    
    args = parser.parse_args()
    
    normalize = not args.no_normalize
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single file
        convert_npy_to_image(args.input, args.output, normalize)
    elif os.path.isdir(args.input):
        # Directory
        convert_directory(args.input, args.output, args.recursive)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return


if __name__ == '__main__':
    main()

