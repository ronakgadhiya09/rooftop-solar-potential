# Rooftop Solar Potential Estimation

An end-to-end deep learning pipeline for **automated rooftop segmentation** and **solar potential estimation** from high-resolution aerial imagery.  

This project presents a deep learning-based pipeline for rooftop solar potential estimation using high-resolution satellite imagery. It combines HRNet with OCR and boundary-aware modules for accurate rooftop segmentation, followed by post-processing and integration with PVGIS to estimate usable rooftop area and annual solar energy yield. The framework is trained in two stages—using the RID dataset for generic feature learning and a custom Indian dataset for local adaptation—supporting city-scale analysis for sustainable urban planning.

---

## Features

- HRNet + OCR backbone with boundary refinement
- Two-stage training: RID pretraining + Indian fine-tuning
- Post-processing with CRF and edge filtering
- Usable area estimation in m²
- Solar energy yield estimation via PVGIS


## References

- HRNet [1], OCR [2], SolarNet [3], SolarNet+ [4]
- PVGIS: [https://ec.europa.eu/jrc/en/pvgis](https://ec.europa.eu/jrc/en/pvgis)

---

## Selected Papers

[1] Wang et al., *Deep High-Resolution Representation Learning*, IEEE TPAMI, 2021.

[2] Yuan et al., *Object-Contextual Representations*, ECCV, 2020.

[3] Qingyu Li, Sebastian Krapf, Lichao Mou, Yilei Shi, Xiao Xiang Zhu. *SolarNet: A convolutional neural network-based framework for rooftop solar potential estimation from aerial imagery Applied Energy*,2024. doi: https://doi.org/10.1016/j.jag.2022.103098

[4] Qingyu Li, Sebastian Krapf, Lichao Mou, Yilei Shi, Xiao Xiang Zhu. *Deep learning-based framework for city-scale rooftop solar potential estimation by consid-ering roof superstructures. Applied Energy*,2024. doi: https://doi.org/10.1016/j.apenergy.2024.123839

[5] Krapf, S.; Bogenrieder, L.; Netzler, F.; Balke, G.; Lienkamp, M. RID—Roof Information Dataset for Computer Vision-Based Photovoltaic Potential Assessment. Remote Sens. 2022, 14, 2299. https://doi.org/10.3390/rs14102299

---

*Feel free to clone, use, or adapt this project.*


