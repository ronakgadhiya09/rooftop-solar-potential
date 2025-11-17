# Rooftop Solar Potential Estimation

- **Code outputs**: Check `solar_potential_estimation_.ipynb`
- **Reports**: Check `Reports/`
- **Presentation**: Check `Presentation/`

## Repository Overview

This repository contains an end-to-end deep learning pipeline for automated rooftop segmentation and solar potential estimation from high-resolution aerial imagery. The system uses HRNet+OCR for rooftop segmentation and a U-Net for tilted roof detection, followed by mask fusion and panel count regression.

### Main Components

- **`Solar_Potential_Estimation/`**: Core codebase containing:
  - **`model/`**: HRNet+OCR segmentation model architecture
  - **`inference/`**: Inference scripts for rooftop segmentation, roof separation, area calculation, and solar potential estimation
  - **`train_script/`**: Training scripts for single and multi-GPU setups
  - **`dataloader/`** & **`datasets/`**: Data loading utilities for INRIA and custom Indian datasets
  - **`configs/`**: Model configuration files (YAML)
  - **`lib/`**: Core utilities for configuration and helper functions
  - **`india_dataset/`**: Custom Indian rooftop dataset with training/validation splits (360 train, 90 val images)
  - **`weights/`**: Pre-trained model weights (download links provided)

- **`solar_potential_estimation_.ipynb`**: Main Jupyter notebook with code outputs and visualizations
- **`Tilted_roofs.ipynb`**: Notebook for tilted roof detection and analysis
- **`requirements.txt`**: Python dependencies (PyTorch, OpenCV, Albumentations, etc.)

### Key Features

- **Rooftop Segmentation**: HRNet+OCR-based binary segmentation for roof footprint detection
- **Tilted Roof Detection**: U-Net model for identifying tilted roof regions
- **Mask Fusion**: Combines base and tilted masks to separate fully usable vs. tilted areas
- **Panel Count Regression**: CNN-based regressor predicting panel counts on normal and tilted areas
- **Solar Potential Calculation**: Integration with PVGIS API for solar energy yield estimation
- **Per-Roof Analysis**: Connected components analysis for individual rooftop identification and area calculation

## Results

### Panel Count Prediction Performance

| Target | R² | RMSE (panels) | MAE (panels) | Acc@±2 | Mean Error |
|--------|----|---------------|--------------|--------|------------|
| Normal panels | 0.876 | 3.62 | 2.48 | 87.9% | -0.12 |
| Tilted panels | 0.742 | 2.31 | 1.47 | 83.6% | +0.08 |

### Key Contributions

- HRNet+OCR-based rooftop segmentation for robust roof footprints
- Trained tilted-roof segmenter with principled mask-fusion strategy to separate fully usable vs. tilted areas
- Two-target regressor that predicts panel counts on normal and tilted areas, producing actionable outputs beyond kWh
