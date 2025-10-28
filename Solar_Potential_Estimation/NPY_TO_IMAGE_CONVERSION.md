# NPY to Image Conversion Script

This script converts NumPy array files (.npy) to PNG images for visualization and inspection.

## Features

- ✅ Convert single .npy files or entire directories
- ✅ Support for binary masks, multi-channel masks, and labeled masks
- ✅ Automatic normalization to 0-255 range
- ✅ Recursive directory searching
- ✅ Detailed output with array information
- ✅ Handles 2D and 3D arrays

## Installation

Make sure you have the required Python packages installed:

```bash
pip install numpy pillow
```

## Usage

### Basic Usage

Convert a single .npy file:
```bash
python convert_npy_to_images.py --input path/to/file.npy
```

Convert all .npy files in a directory:
```bash
python convert_npy_to_images.py --input path/to/directory
```

### Advanced Options

**Specify output directory:**
```bash
python convert_npy_to_images.py --input masks/ --output converted_images/
```

**Convert recursively (all subdirectories):**
```bash
python convert_npy_to_images.py --input india_dataset/ --recursive
```

**Disable normalization (preserve original values):**
```bash
python convert_npy_to_images.py --input masks/ --no-normalize
```

### Examples for Your Dataset

1. **Convert train masks:**
   ```bash
   python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks --output train_mask_images
   ```

2. **Convert validation masks:**
   ```bash
   python convert_npy_to_images.py --input india_dataset/twoChannels_in/val/masks --output val_mask_images
   ```

3. **Convert test masks:**
   ```bash
   python convert_npy_to_images.py --input test_crowd/masks --output test_mask_images
   ```

4. **Convert all masks at once:**
   ```bash
   python convert_npy_to_images.py --input india_dataset/ --recursive --output all_converted_masks
   ```

## Arguments

- `--input` / `-i`: Path to .npy file or directory containing .npy files (required)
- `--output` / `-o`: Output directory for converted images (optional, defaults to input location)
- `--recursive` / `-r`: Search for .npy files recursively in subdirectories
- `--no-normalize`: Disable normalization to 0-255 range (preserves original values)

## How It Works

1. **Loads the .npy file** using `numpy.load()`
2. **Handles different dimensions:**
   - 2D arrays: Grayscale images
   - 3D arrays with depth 1: Squeeze to 2D
   - 3D arrays with 3 channels: RGB image
3. **Normalizes values** to 0-255 range (unless `--no-normalize` is used)
4. **Converts to PIL Image** and saves as PNG

## Output

The script provides detailed information for each file:
- File path being processed
- Array shape and data type
- Min and max values
- Output location

Example output:
```
Processing: india_dataset/twoChannels_in/train/masks/1_1.npy
  Shape: (512, 512)
  Dtype: int64
  Min: 0, Max: 1
  Saved to: india_dataset/twoChannels_in/train/masks/images/1_1_converted.png
```

## Notes

- Binary masks (0s and 1s) are automatically scaled to 0-255 for better visualization
- The output filename will be `{original_name}_converted.png`
- If no output directory is specified, images are saved in the same directory as the input files

