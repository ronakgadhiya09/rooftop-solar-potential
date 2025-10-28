# Virtual Environment Setup Guide

This project uses Python 3 with a virtual environment for dependency management.

## Quick Start

### 1. Create and Setup Virtual Environment (Automated)

Run the setup script:
```bash
bash setup_venv.sh
```

This will:
- Create a virtual environment named `venv`
- Activate it
- Upgrade pip
- Install all dependencies from `requirements.txt`

### 2. Manual Setup

If you prefer to set it up manually:

#### Create virtual environment
```bash
python3 -m venv venv
```

#### Activate virtual environment
```bash
source venv/bin/activate
```

#### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Using the Virtual Environment

### Activate the environment
```bash
source venv/bin/activate
```

When activated, your terminal prompt will show `(venv)` at the beginning.

### Deactivate the environment
```bash
deactivate
```

### Run scripts with the environment activated

Example - Convert npy to images:
```bash
source venv/bin/activate
python convert_npy_to_images.py --input india_dataset/twoChannels_in/train/masks --output train_converted_images
```

Example - Run training:
```bash
source venv/bin/activate
python train_script/i_train_single_gpu.py --cfg configs/inria_hrnet_ocr.yaml
```

Example - Run inference:
```bash
source venv/bin/activate
python inference/i_inference.py --checkpoint i_outputs/epoch_99.pth --image path/to/image.png
```

## Dependencies

The project requires the following packages:
- **torch** - PyTorch deep learning framework
- **torchvision** - Computer vision utilities for PyTorch
- **numpy** - Numerical computations
- **pillow** - Image processing
- **opencv-python** - Computer vision library
- **pandas** - Data manipulation
- **albumentations** - Image augmentation
- **scikit-learn** - Machine learning utilities
- **scikit-image** - Image processing
- **pytorch-msssim** - SSIM loss for PyTorch
- **yacs** - Configuration management
- **matplotlib** - Plotting and visualization
- **requests** - HTTP requests

## Troubleshooting

### CUDA issues (GPU support)
If you have a CUDA-enabled GPU, install PyTorch with CUDA support:
```bash
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Missing dependencies
If you encounter import errors, reinstall requirements:
```bash
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

### Environment not activating
Make sure the venv directory exists and try recreating it:
```bash
rm -rf venv
bash setup_venv.sh
```

## IDE Setup (Optional)

### PyCharm
1. File → Settings → Project → Python Interpreter
2. Click the gear icon → Add
3. Select "Existing environment"
4. Browse to `venv/bin/python`
5. Click OK

### VS Code
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `./venv/bin/python`

## Notes

- Always activate the virtual environment before running any scripts
- The `venv` folder is excluded from version control (should be in .gitignore)
- If you add new dependencies, update `requirements.txt`:
  ```bash
  source venv/bin/activate
  pip freeze > requirements.txt
  ```

