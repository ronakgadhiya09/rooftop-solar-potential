# Tilted Roof Training Dataset

This document describes the dataset prepared for training a model to identify tilted roofs.

## Dataset Overview

### Location
- Base directory: `india_dataset/twoChannels_in/train/`
- Images: `images/` (360 PNG files)
- Tilted masks: `tilted_masks/` (90 NPY files)

### Dataset Composition

#### Positive Examples (Tilted Roofs)
- **Count**: 70 images
- **Source**: Converted from JSON polygon annotations in `json_masks/`
- **Format**: Binary masks (0s and 1s) where 1 indicates tilted roof areas
- **Examples**: `1_4.npy`, `223_2.npy`, `20_1.npy`, etc.

#### Negative Examples (Non-Tilted Roofs)
- **Count**: 20 images  
- **Source**: Images without JSON annotations (no tilted roofs detected)
- **Format**: Completely black masks (all zeros)
- **Examples**: `10_1.npy`, `11_2.npy`, `13_3.npy`, etc.

#### Total Training Set
- **Total images**: 90 (70 positive + 20 negative)
- **Class distribution**: 
  - Tilted roofs: ~77.8%
  - Non-tilted roofs: ~22.2%

## Files Created

### NPY Masks
```
india_dataset/twoChannels_in/train/tilted_masks/
├── [70 positive masks from JSON]
│   ├── 1_4.npy          (contains roof pixels)
│   ├── 223_2.npy        (contains roof pixels)
│   └── ...
└── [20 negative masks]
    ├── 10_1.npy         (all zeros - black mask)
    ├── 11_2.npy         (all zeros - black mask)
    └── ...
```

### Visualizations
```
india_dataset/twoChannels_in/train/tilted_masks_visualized/
└── [70 PNG visualizations of positive examples]
    ├── 1_4_converted.png
    ├── 223_2_converted.png
    └── ...
```

## Dataset Statistics

### Positive Examples Coverage
- Average coverage varies per image
- Example coverages:
  - `20_1.npy`: 27.0% (70,907 pixels)
  - `223_2.npy`: 37.9% (99,425 pixels)
  - `1_4.npy`: 10.4% (27,325 pixels)

### Negative Examples
- All masks are completely black (0% coverage)
- Used as negative examples for binary classification

## Usage for Training

### Loading the Dataset

```python
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class TiltedRoofDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Get all mask files
        self.mask_files = sorted(list(Path(masks_dir).glob('*.npy')))
        
    def __len__(self):
        return len(self.mask_files)
    
    def __getitem__(self, idx):
        mask_path = self.mask_files[idx]
        image_name = mask_path.stem + '.png'
        image_path = self.images_dir / image_name
        
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Load mask
        mask = np.load(mask_path)
        
        # Determine label (1 if has roof pixels, 0 if all black)
        label = 1 if mask.max() > 0 else 0
        
        return {
            'image': image,
            'mask': mask,
            'label': label,
            'name': mask_path.stem
        }

# Usage
dataset = TiltedRoofDataset(
    'india_dataset/twoChannels_in/train/images',
    'india_dataset/twoChannels_in/train/tilted_masks'
)
```

### Checking Label Distribution

```python
# Count positive and negative examples
positive_count = 0
negative_count = 0

for mask_file in Path('india_dataset/twoChannels_in/train/tilted_masks').glob('*.npy'):
    mask = np.load(mask_file)
    if mask.max() > 0:
        positive_count += 1
    else:
        negative_count += 1

print(f"Positive (tilted): {positive_count}")
print(f"Negative (not tilted): {negative_count}")
```

## Scripts Available

1. **`convert_json_to_npy.py`** - Convert JSON annotations to NPY masks
2. **`batch_convert_json_masks.py`** - Batch convert all JSON files
3. **`create_negative_masks.py`** - Create negative examples (black masks)
4. **`convert_npy_to_images.py`** - Visualize masks as PNG images

## Dataset Balance

**Current Distribution:**
- Positive: 70 examples (77.8%)
- Negative: 20 examples (22.2%)

**Note:** The dataset is slightly imbalanced. You may want to:
1. Add more negative examples using `create_negative_masks.py --num 50`
2. Use class weighting in your loss function
3. Use data augmentation to balance the classes

## Adding More Negative Examples

If you need more negative examples:

```bash
source venv/bin/activate
python create_negative_masks.py --num 50  # Add 50 more negative examples
```

This will create black masks for additional images that don't have tilted roofs.

## Next Steps

1. ✅ Dataset prepared: 90 masks (70 positive, 20 negative)
2. 📝 Create DataLoader for training
3. 🏗️ Set up model architecture for binary segmentation/classification
4. 🚀 Train model to detect tilted roofs
5. 📊 Evaluate on validation set

## Verification

To verify a negative mask is all zeros:
```python
import numpy as np
mask = np.load('india_dataset/twoChannels_in/train/tilted_masks/10_1.npy')
assert mask.sum() == 0, "Negative mask should be all zeros"
assert mask.max() == 0, "Negative mask should have max value 0"
```

