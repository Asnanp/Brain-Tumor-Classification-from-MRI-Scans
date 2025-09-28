# 🧠 Brain Tumor Classification from MRI Scans  
### *A Reproducible Deep Learning Pipeline for Multi-Class Medical Image Diagnosis*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-7k%2B%20MRI-blue)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

> **Early and accurate brain tumor classification can significantly improve patient outcomes.**  
> This project implements a **robust, reproducible deep learning pipeline** for 4-class classification of brain MRI scans using modern PyTorch best practices.

---

## 🔬 Problem Statement
Automated classification of brain tumors from MRI scans into:
- **Glioma** (aggressive, often malignant)
- **Meningioma** (typically benign)
- **Pituitary Adenoma**
- **No Tumor**

Accurate differentiation is critical: treatment protocols vary drastically by tumor type, and misclassification can lead to delayed or incorrect interventions.

---

## 📊 Performance Highlights
| Metric                | Value     |
|-----------------------|-----------|
| **Test Accuracy**     | 92.45%    |
| **Pituitary Recall**  | 100%      |
| **Glioma Recall**     | 85%       |
| **Model**             | ResNet18 (Transfer Learning) |
| **Dataset Size**      | 7,023 MRI images |

![Confusion Matrix](assets/confusion_matrix.png)  
*Confusion matrix on held-out test set — note near-perfect pituitary classification and room for improvement in glioma/meningioma distinction.*

---
<img width="649" height="545" alt="download" src="https://github.com/user-attachments/assets/628bccad-a0f0-4135-a618-733c40cce890" />

## 🛠️ Technical Stack
- **Framework**: PyTorch 2.0+ with `torchvision`
- **Model**: ResNet18 (ImageNet-pretrained), with full fine-tuning support
- **Preprocessing**: 
  - Dynamic resizing (224×224)
  - Grayscale → RGB conversion for pretrained compatibility
  - Medical-aware normalization (ImageNet stats)
- **Augmentation**: 
  - Random horizontal flip (p=0.3)
  - Random rotation (±10°)
- **Training**: 
  - Class-balanced sampling (optional)
  - Early stopping & model checkpointing
  - GPU-accelerated (CUDA support)
- **Evaluation**: Per-class precision/recall/F1, confusion matrix, classification report

---

## 🚀 Quick Start
1. Clone & Install
```
git clone https://github.com/AsnanP/Brain-Tumor-Classification-from-MRI-Scans.git
cd Brain-Tumor-Classification-from-MRI-Scans
pip install -r requirements.txt
```
2. Organize Data
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```
3. Train
```python src/train.py --data_dir ./data --epochs 20 --batch_size 32 --lr 0.001```
4. Evaluate
```python src/evaluate.py --model_path best_model.pth --data_dir ./data```
<img width="990" height="372" alt="download" src="https://github.com/user-attachments/assets/17840cb7-b0a6-47e4-81a7-5b0200a948ae" />


### 🔍 Key Insights & Challenges
Class Confusion: Glioma and meningioma show cross-prediction due to overlapping visual features in axial MRI slices.
Data Quality: Removed mislabeled samples from SARTAJ subset (as noted in dataset source).
Medical Ethics: Model designed as decision-support tool, not diagnostic replacement.
Generalization: High accuracy on notumor and pituitary suggests strong feature learning for distinct pathologies.
### 📈 Future Work
Grad-CAM integration for tumor localization visualization
3D CNN for volumetric MRI analysis (using full scan series)
Uncertainty quantification (Monte Carlo Dropout)
Federated learning setup for privacy-preserving multi-hospital training
ONNX export + TorchServe deployment
# 📚 References
WHO Classification of Tumours of the Central Nervous System
He et al., Deep Residual Learning for Image Recognition (ResNet)
Dataset: Brain Tumor MRI Dataset (Kaggle)
# 🤝 Contributing
Contributions welcome! Especially:

Improved preprocessing (skull-stripping, bias correction)
New model architectures (ViT, ConvNeXt)
Clinical validation metrics (sensitivity/specificity per tumor grade)
