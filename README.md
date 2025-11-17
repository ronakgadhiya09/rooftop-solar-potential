# Rooftop Solar Potential Estimation

- **Code outputs**: Check `solar_potential_estimation_.ipynb`
- **Reports**: Check `Reports/`
- **Presentation**: Check `Presentation/`

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
