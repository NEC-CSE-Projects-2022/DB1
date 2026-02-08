# Learning Ridge Structures: Adaptive CNN-Based Local Orientation and Frequency Estimation for Fingerprints

📂 **Project Resources (Paper, Dataset, Outputs, etc.)**  
🔗 https://drive.google.com/drive/folders/1qTMYMxM7c4SEV0a4bTkG3xozVDYKyKPz?usp=sharing

---

## 📌 Overview
Fingerprint recognition is one of the most reliable biometric identification techniques, where **ridge orientation** and **ridge frequency** play a crucial role in enhancement, minutiae extraction, and matching.  
This project presents an **adaptive CNN-based end-to-end framework** to simultaneously estimate **local ridge orientation and ridge frequency** from fingerprint images, even under noisy or low-quality conditions.

The work is based on the research paper presented at **ICAICCIT 2025**, focusing on combining **domain-specific preprocessing** with **deep learning** to improve robustness and accuracy.

---

## 🧠 Key Contributions
- Adaptive CNN architecture for **joint ridge orientation and frequency estimation**
- Integration of **XSFFE** and **SNFFE** preprocessing techniques
- **Pixel-wise dense supervision** for fine-grained ridge learning
- Strong robustness to **noise, smudges, low contrast, and partial fingerprints**
- Achieved **MAPE of 4.58%** for ridge frequency estimation
- Suitable for **real-time applications**

---

## 🏗️ System Architecture
The system processes raw fingerprint images through the following stages:

1. **Preprocessing**
   - XSFFE (Extended Synthetic Fingerprint Feature Extraction)
   - SNFFE (Synthetic Normalized Fingerprint Feature Extraction)
2. **Segmentation**
   - Foreground (ROI) extraction to remove background noise
3. **Adaptive CNN Model**
   - Learns pixel-wise ridge orientation and frequency
4. **Post-processing**
   - Gaussian smoothing
   - Missing value filling
   - Orientation & frequency map reconstruction

---

## 🧪 Dataset
- **FFE Dataset**
  - Designed for evaluating ridge orientation, ridge frequency, and foreground detection
  - Includes **Good** and **Bad (degraded)** quality fingerprint images
  - Supports benchmarking of robustness under noisy conditions

---

## ⚙️ Model Details
- **Convolutional Layers:** 6 (3×3 kernels)
- **Activation Function:** ReLU
- **Optimizer:** Adam
- **Batch Size:** 32
- **Epochs:** 100
- **Training Strategy:** Pixel-wise dense supervision with data augmentation

---

## 📊 Evaluation Results
### Accuracy (MAPE on FFE Dataset)
- **SNFFE:**  
  - Good Images: **4.58%**  
  - Bad Images: **5.02%**
- Outperformed traditional CNN, regression, and orientation-vector models

### Execution Time
- **SNFFE:** 0.012 seconds per fingerprint  
- Demonstrates strong suitability for **real-time systems**

---

## 🔍 Features Extracted
- Local ridge **orientation maps**
- Local ridge **frequency maps**
- Fused **orientation + frequency vector fields**
- X-signature extraction for ridge pattern analysis

---

## 🚀 Applications
- Fingerprint image enhancement
- Minutiae extraction
- Automated Fingerprint Identification Systems (AFIS)
- Real-time biometric authentication
- Research in fingerprint biometrics and forensic analysis

---

## ⚠️ Limitations
- Limited evaluation on latent or extremely degraded fingerprints
- Tested mainly on the FFE dataset
- Ablation studies for XSFFE and SNFFE are limited

---

## 🔮 Future Scope
- Integration into full end-to-end fingerprint recognition pipelines
- Evaluation on cross-sensor and cross-population datasets
- Extension to latent fingerprint enhancement
- Adversarial robustness against spoofing attacks
- Detailed ablation and statistical validation studies

---

## 📄 Publication
**Learning Ridge Structures: Adaptive CNN-Based Local Orientation and Frequency Estimation for Fingerprints**  
Presented at **3rd International Conference on Advances in Computation, Communication and Information Technology (ICAICCIT 2025)**  
© IEEE 2025

---

## 👨‍💻 Authors
- Jashwanth Atmakuri  
- Dr. Rizwana Syed  
- Farzan Basha Shaik Inkollu  
- Leela Krishna Yalagala  
- Rama Sundari  
- Sesha Bhargavi Velagaleti  
- Venkata Narasimhareddy Kandi

---

## 📜 License
This project is intended for **academic and research purposes**.  
Please cite the original paper if you use this work.
