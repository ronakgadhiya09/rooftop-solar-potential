# Tilted Roof Training Notebook Guide

This guide explains the training cells added to the Jupyter notebook for training a tilted roof detection model.

## Notebook Structure

The training script is organized into 11 cells:

### Cell 1: Import Libraries
- Imports all necessary libraries (PyTorch, NumPy, PIL, etc.)
- Sets up paths for model imports
- Checks CUDA availability

### Cell 2: Dataset Class
- `TiltedRoofDataset` class that loads images and masks from the `tilted_masks` directory
- Supports data augmentation for training set
- Automatically handles mask format conversions

### Cell 3: Loss Functions
- **FocalLoss**: Handles class imbalance (important for tilted roofs which are ~17% of pixels)
- **DiceLoss**: Measures overlap between prediction and ground truth

### Cell 4: Configuration
- Creates model configuration matching HRNet-OCR architecture
- Sets training hyperparameters (batch size, learning rate, epochs)
- Same architecture as your main roof segmentation model

### Cell 5: Model Initialization
- Initializes HRNet-OCR model
- Loads pre-trained weights if available (from main roof segmentation)
- Reinitializes classifier head for binary segmentation

### Cell 6: DataLoaders
- Creates train/validation split (80/20)
- Sets up PyTorch DataLoaders with proper batching

### Cell 7: Training Functions
- `train_epoch()`: Training loop for one epoch
- `validate()`: Validation with metrics calculation (Precision, Recall, F1, IoU)

### Cell 8: Training Loop
- Main training loop for specified number of epochs
- Saves best model based on IoU score
- Saves checkpoints every 10 epochs
- Tracks training history

### Cell 9: Training History Visualization
- Plots training curves for loss, metrics
- Saves plots to output directory

### Cell 10: Prediction on Training Sample
- Loads best trained model
- Makes prediction on a random training sample
- Calculates per-sample metrics

### Cell 11: Visualization
- Comprehensive visualization comparing predicted vs actual masks
- Shows:
  - Original image
  - Actual mask
  - Predicted mask
  - Prediction probability heatmap
  - Overlays on original image
  - Comparison mask (TP/FP/FN visualization)

## Usage Instructions

### In Google Colab:

1. **Update Paths** (Cell 6):
   ```python
   BASE_DIR = '/content/drive/MyDrive/your_project_folder'
   IMAGES_DIR = os.path.join(BASE_DIR, 'india_dataset/twoChannels_in/train/images')
   MASKS_DIR = os.path.join(BASE_DIR, 'india_dataset/twoChannels_in/train/tilted_masks')
   ```

2. **Adjust Training Parameters** (Cell 4):
   ```python
   cfg.SOLVER.BATCH_SIZE = 8  # Adjust based on GPU memory
   cfg.SOLVER.BASE_LR = 0.0001  # Learning rate
   cfg.SOLVER.MAX_EPOCHES = 50  # Number of epochs
   ```

3. **Run Cells Sequentially**: Execute cells 1-11 in order

## Model Architecture

Uses the same HRNet-OCR architecture as your main model:
- **High-Resolution Network (HRNet)**: Maintains high-resolution representations
- **Object Contextual Representation (OCR)**: Captures object context
- **Binary Segmentation Head**: Outputs single channel for tilted roof detection

## Training Details

- **Loss Function**: Combined Focal Loss + Dice Loss
  - Focal Loss: α=0.25, γ=2.0 (handles class imbalance)
  - Dice Loss: Measures pixel overlap
  - Total Loss = Focal Loss + Dice Loss

- **Optimizer**: AdamW
  - Learning rate: 0.0001
  - Weight decay: 0.0001

- **Scheduler**: CosineAnnealingLR
  - Gradually reduces learning rate

- **Data Augmentation**:
  - Horizontal/Vertical flips
  - Random rotation
  - Brightness/contrast adjustment
  - Elastic transform
  - Gaussian blur
  - Cutout

## Output Files

All outputs are saved in `tilted_roof_outputs/`:
- `best_tilted_roof_model.pth`: Best model checkpoint
- `checkpoint_epoch_X.pth`: Checkpoints every 10 epochs
- `training_history.png`: Training curves
- `prediction_comparison_[sample_name].png`: Prediction visualization
- `comparison_mask_[sample_name].png`: TP/FP/FN visualization

## Evaluation Metrics

The model is evaluated using:
- **Precision**: Correct positive predictions / All positive predictions
- **Recall**: Correct positive predictions / All actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: Pixel overlap measure

## Tips

1. **GPU Memory**: If you get OOM errors, reduce `BATCH_SIZE` to 4 or 2

2. **Training Time**: On Colab T4 GPU, ~50 epochs takes ~1-2 hours for 90 samples

3. **Early Stopping**: The best model is saved automatically based on validation IoU

4. **Hyperparameter Tuning**: 
   - Increase epochs if validation metrics are still improving
   - Adjust learning rate if training is unstable
   - Increase batch size if you have more GPU memory

5. **Adding More Data**: Simply add more .npy files to `tilted_masks/` directory and retrain

## Testing on Specific Samples

To test on a specific image instead of random selection (Cell 10):

```python
# Replace random selection with specific sample
sample_name = "223_2"  # Your desired sample name
sample_idx = [i for i, s in enumerate(train_dataset) if s['name'] == sample_name][0]
sample = train_dataset[sample_idx]
```

## Troubleshooting

**Import Errors**: Make sure model files are in the correct path
```python
# Adjust sys.path in Cell 1 if needed
sys.path.append('/path/to/your/model/directory')
```

**File Not Found**: Update paths in Cell 6 to match your directory structure

**CUDA Out of Memory**: 
- Reduce batch size
- Reduce image size (modify transforms)
- Use gradient accumulation

**Poor Performance**:
- Train for more epochs
- Adjust learning rate
- Add more training data (especially negative examples)
- Try different loss weights

## Next Steps

After training:
1. Evaluate on validation set
2. Test on unseen images
3. Fine-tune hyperparameters if needed
4. Integrate with main inference pipeline

