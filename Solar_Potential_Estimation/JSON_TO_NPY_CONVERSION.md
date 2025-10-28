# JSON to NPY Conversion Script

This script converts JSON polygon annotation files to NumPy mask arrays (.npy files).

## What It Does

The script reads JSON files containing polygon annotations (like those from image annotation tools) and converts them into binary mask arrays suitable for machine learning training.

### Input: JSON Format
```json
[
    {
        "content": [
            {"x": 0, "y": 150.21},
            {"x": 296.79, "y": 37.02},
            ...
        ],
        "contentType": "polygon",
        ...
    },
    ...
]
```

### Output: NPY Binary Mask
- Binary mask array (0s and 1s)
- Shape: (height, width) matching the image dimensions
- Format: numpy .npy file

## Usage

### Single File Conversion

Convert a single JSON file to .npy:
```bash
source venv/bin/activate
python convert_json_to_npy.py --input india_dataset/twoChannels_in/train/masks/223_2.json --auto-size
```

With custom output path:
```bash
python convert_json_to_npy.py --input 223_2.json --output custom_mask.npy --auto-size
```

With manual size specification:
```bash
python convert_json_to_npy.py --input 223_2.json --size 512,512
```

### Directory Conversion

Convert all JSON files in a directory:
```bash
python convert_json_to_npy.py --input india_dataset/twoChannels_in/train/masks --auto-size
```

## Options

- `--input` / `-i`: Input JSON file or directory (required)
- `--output` / `-o`: Output .npy file path (optional, defaults to same name as input with .npy extension)
- `--size width,height`: Manual image size specification (e.g., 512,512)
- `--auto-size`: Automatically detect image size from corresponding image files in the images directory

## How It Works

1. **Reads JSON annotations**: Parses polygon coordinates from the JSON file
2. **Finds image dimensions**: 
   - If `--auto-size` is used, searches for the corresponding image file (e.g., `223_2.png`)
   - If `--size` is provided, uses that size
   - Otherwise defaults to 512x512
3. **Creates binary mask**: Uses OpenCV to fill polygons on the mask
4. **Saves as .npy**: Outputs a binary numpy array (0s and 1s)

## Example Output

```
Found corresponding image: india_dataset/twoChannels_in/train/images/223_2 тай щь (512, 512)
Created mask shape: (512, 512)
Mask dtype: uint8
Min: 0, Max: 1
Non-zero pixels: 79629 / 262144 (30.4%)
Saved mask to: india_dataset/twoChannels_in/train/masks/223_2.npy
```

## Verification

After conversion, you can verify the mask by converting it to an image:

```bash
# Convert npy to image to visualize
python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks/223_2.npy --output visualization
```

## File Organization

The script expects this directory structure:
```
dataset/
├── train/
│   ├── images/
│   │   └── 223_2.png          # Original image
│   └── masks/
│       ├── 223_2.json         # JSON annotations (input)
│       └── 223_2.npy          # Binary mask (output)
```

If your JSON files have corresponding images in an `images` directory at the same level, the `--auto-size` flag will automatically detect the correct dimensions.

## Notes

- The mask format matches your existing .npy masks: binary (0 or 1)
- Polygon coordinates are automatically converted to integers
- The script handles multiple polygons in a single JSON file
- Existing .npy files will be overwritten if you convert the same file

## Integration with Training Pipeline

The generated .npy files can be used directly with your training pipeline:

```python
# In your dataset loader
mask = np.load(mask_path)  # Shape: (H, W), dtype: uint8, values: 0 or 1
```

This matches the format expected by `datasets/inria.py` and `dataloader/inria.py`.

