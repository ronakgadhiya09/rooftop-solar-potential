#!/bin/bash
# Example usage of the convert_npy_to_images.py script

echo "=== Convert .npy files to PNG images ==="
echo ""

# Example 1: Convert a single file
echo "Example 1: Convert a single .npy file"
echo "python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks/1_1.npy"
echo ""

# Example 2: Convert all .npy files in train masks directory
echo "Example 2: Convert all .npy files in train/masks directory"
echo "python convert_nAGy_to_images.py --input india_dataset/twoChannels_in/train/masks"
echo ""

# Example 3: Convert all .npy files with custom output directory
echo "Example 3: Convert with custom output directory"
echo "python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks --output train_converted_images"
echo ""

# Example 4: Convert val masks to images
echo "Example 4: Convert val masks to images"
echo "python convert_npy_to_images.py --input india_dataset/twoChannels_in/val/masks --output val_converted_images"
echo ""

# Example 5: Convert recursively (all directories)
echo "Example 5: Convert all .npy files recursively"
echo "python convert_npy_to_images.py --input india_dataset --recursive --output converted_all_masks"
echo ""

echo "Note: Make sure you have numpy and Pillow installed:"
echo "pip install numpy pillow"

