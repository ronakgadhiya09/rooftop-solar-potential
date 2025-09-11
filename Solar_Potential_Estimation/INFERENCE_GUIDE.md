# Solar Potential Estimation - Inference Guide

This guide provides comprehensive instructions for running the inference pipeline to estimate rooftop solar potential from aerial imagery.

## Table of Contents
1. [Setup](#setup)
2. [Downloading Datasets](#downloading-datasets)
3. [Downloading Pre-trained Weights](#downloading-pre-trained-weights)
4. [Running Inference](#running-inference)
5. [Calculating Solar Potential](#calculating-solar-potential)
6. [Output Files](#output-files)
7. [Troubleshooting](#troubleshooting)

## Setup

### Environment Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install torch torchvision
   pip install yacs albumentations opencv-python matplotlib pandas pillow scikit-image
   pip install pytorch-msssim opencv-contrib-python requests
   pip install gdown  # For downloading datasets
   ```

### Directory Structure

Ensure you have the following directory structure:
```
Solar_Potential_Estimation/
├── configs/
│   └── inria_hrnet_ocr.yaml
├── datasets/
│   └── inria.py
├── inference/
│   ├── get_solar_data.py
│   ├── i_inference.py
│   └── i_roof_separation_with_area.py
├── lib/
│   └── config/
│       └── config.py
├── model/
│   └── seg_hrnet_ocr.py
├── i_outputs/
│   └── epoch_99.pth  # Pre-trained weights
├── test_crowd/
│   ├── images/       # Input images
│   ├── masks/        # Intermediate masks
│   ├── pred/         # Prediction outputs
│   └── original/     # Final results
│       ├── components/
│       ├── roof_data/
│       ├── total_pon/
│       └── com_bin/
└── solar_power_json/
```

Create the required directories:
```bash
mkdir -p Solar_Potential_Estimation/test_crowd/images
mkdir -p Solar_Potential_Estimation/test_crowd/masks
mkdir -p Solar_Potential_Estimation/test_crowd/pred
mkdir -p Solar_Potential_Estimation/test_crowd/original/components
mkdir -p Solar_Potential_Estimation/test_crowd/original/roof_data
mkdir -p Solar_Potential_Estimation/test_crowd/original/total_pon
mkdir -p Solar_Potential_Estimation/test_crowd/original/com_bin
mkdir -p Solar_Potential_Estimation/solar_power_json
mkdir -p Solar_Potential_Estimation/i_outputs
```

## Downloading Datasets

### Custom Indian Dataset
```bash
gdown --folder https://drive.google.com/drive/folders/1t7Aa_YWsMdeAfqmiik86U_zewLbeazK7 -O Solar_Potential_Estimation/india_dataset/twoChannels_in
```

After downloading, extract the RAR file:
```bash
unrar x Solar_Potential_Estimation/india_dataset/twoChannels_in/twoChannels_in.rar Solar_Potential_Estimation/india_dataset/
```

### RID Dataset (Optional for Inference)
The RID dataset is required only for training. For inference, you can skip this step.

To download the RID dataset:
1. Visit https://mediatum.ub.tum.de/1655470
2. Accept the terms of use
3. Download the dataset
4. Extract to `Solar_Potential_Estimation/inria_dataset/twoChannels_in`

## Downloading Pre-trained Weights

Download the pre-trained weights for inference:
```bash
gdown --folder https://drive.google.com/drive/folders/1RmPxBfePZctk_RLSwMcZFcjqx4HxDiI7 -O Solar_Potential_Estimation/i_outputs
```

## Running Inference

### Preparing Input Images

Place your aerial images in the `Solar_Potential_Estimation/test_crowd/images/` directory. The images should be high-resolution aerial photographs of rooftops.

For testing, you can copy a sample image from the dataset:
```bash
cp Solar_Potential_Estimation/india_dataset/twoChannels_in/val/images/25_6.png Solar_Potential_Estimation/test_crowd/images/
```

### Running the Inference Script

To run inference on a single image:
```bash
python Solar_Potential_Estimation/inference/i_inference.py \
  --checkpoint Solar_Potential_Estimation/i_outputs/epoch_99.pth \
  --cfg Solar_Potential_Estimation/configs/inria_hrnet_ocr.yaml \
  --image Solar_Potential_Estimation/test_crowd/images/your_image.png \
  --output Solar_Potential_Estimation/test_crowd/pred
```

To run inference on all images in a directory:
```bash
python Solar_Potential_Estimation/inference/i_inference.py \
  --checkpoint Solar_Potential_Estimation/i_outputs/epoch_99.pth \
  --cfg Solar_Potential_Estimation/configs/inria_hrnet_ocr.yaml \
  --dir Solar_Potential_Estimation/test_crowd/images \
  --output Solar_Potential_Estimation/test_crowd/pred
```

### Inference Output

The inference script generates two files for each input image:
1. `{image_name}.npy`: Binary mask of the predicted rooftops (numpy array)
2. `{image_name}_.png`: Visualization of the original image and the predicted mask

These files are saved in the output directory specified by the `--output` parameter.

## Calculating Solar Potential

After running inference, copy the generated `.npy` files to the masks directory:
```bash
cp Solar_Potential_Estimation/test_crowd/pred/*.npy Solar_Potential_Estimation/test_crowd/masks/
```

Then run the roof separation and solar potential calculation script:
```bash
python Solar_Potential_Estimation/inference/i_roof_separation_with_area.py
```

This script:
1. Identifies individual rooftops using connected components analysis
2. Calculates the area of each rooftop
3. Estimates the number of solar panels that can be installed
4. Calculates the solar potential using the PVGIS API
5. Generates visualizations and data files

## Output Files

After running the complete pipeline, you will find the following output files:

### Inference Outputs (`test_crowd/pred/`)
- `{image_name}.npy`: Binary mask of the predicted rooftops
- `{image_name}_.png`: Visualization of the original image and the predicted mask

### Solar Potential Outputs (`test_crowd/original/`)

#### Components Directory (`components/`)
- `{image_name}.png`: Visualization of the identified rooftops, with each rooftop assigned a unique color and ID

#### Roof Data Directory (`roof_data/`)
- `{image_name}.csv`: CSV file containing detailed information about each rooftop:
  - `Roof_ID`: Unique identifier for each rooftop
  - `Roof_Area`: Area of the rooftop in pixels
  - `Net_Usable_Area`: Usable area of the rooftop in pixels
  - `Real_Area`: Actual area of the rooftop in square meters
  - `Panels`: Number of solar panels that can be installed
  - `Solar_potential_per_year`: Estimated solar energy production in kWh per year

#### Total Potential Directory (`total_pon/`)
- `{image_name}.txt`: Total solar potential for the entire image in kWh per year

#### Component Binary Directory (`com_bin/`)
- `{image_name}.npy`: Numpy array with labeled rooftops (each rooftop has a unique ID)

### PVGIS Data (`solar_power_json/`)
- `solar_data.json`: Raw data from the PVGIS API with detailed solar radiation information

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   - Ensure your virtual environment is activated
   - Check that all required packages are installed
   - Make sure you are running the scripts from the correct directory

2. **CUDA errors**:
   - The code automatically detects if CUDA is available and falls back to CPU if not
   - For faster inference, use a machine with a CUDA-compatible GPU

3. **API errors in solar potential calculation**:
   - Check your internet connection
   - The PVGIS API might be temporarily unavailable
   - You can modify the coordinates in `get_solar_data.py` to match your region

4. **Missing directories**:
   - Ensure all required directories are created before running the scripts
   - The scripts will create some directories automatically, but it is better to create them in advance

### Getting Help

If you encounter issues not covered in this guide, check:
- The original repository documentation
- The code comments for additional information
- The references mentioned in the README.md file

## References

- HRNet: Wang et al., *Deep High-Resolution Representation Learning*, IEEE TPAMI, 2021
- OCR: Yuan et al., *Object-Contextual Representations*, ECCV, 2020
- PVGIS API: https://ec.europa.eu/jrc/en/pvgis
- RID Dataset: Krapf et al., *RID—Roof Information Dataset for Computer Vision-Based Photovoltaic Potential Assessment*, Remote Sensing, 2022
