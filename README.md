# Team 5 – Learning Ridge Structures using Adaptive CNN

## Team Info
22471A05L3- Jashwanth Atmakuri  
Work Done: CNN model design, preprocessing implementation (XSFFE and SNFFE), training, evaluation, documentation

22471A05P1- Farzan Basha Shaik Inkollu  
Work Done: Dataset analysis, preprocessing support, experimental validation, result analysis

22471A05P2- Leela Krishna Yalagala  
Work Done: Model testing, performance comparison, execution time analysis

Dr. Rizwana Syed (Guide)  
Work Done: Project supervision, research guidance, methodology design

---

## Abstract
Fingerprint recognition relies heavily on accurate estimation of ridge orientation and ridge frequency. Traditional handcrafted methods often fail under noisy or low-quality fingerprint images. This project proposes an adaptive CNN-based end-to-end framework that simultaneously estimates local ridge orientation and ridge frequency using pixel-wise dense supervision. By integrating XSFFE and SNFFE preprocessing techniques, the system achieves strong robustness to noise, smudges, and partial fingerprints, delivering improved accuracy and real-time performance.

---

## Paper Reference (Inspiration)
👉 Learning Ridge Structures: Adaptive CNN-Based Local Orientation and Frequency Estimation for Fingerprints  
Rizwana Syed et al., IEEE ICAICCIT 2025  
Original IEEE conference paper used as inspiration and technical foundation for this project.

---

## Our Improvement Over Existing Paper
- Implemented a complete end-to-end pipeline including preprocessing, segmentation, CNN modeling, and post-processing  
- Enhanced robustness through XSFFE and SNFFE synthetic feature extraction  
- Achieved lower Mean Absolute Percentage Error (MAPE = 4.58 percent) for frequency estimation  
- Optimized execution time to 0.012 seconds per fingerprint, making the system suitable for real-time applications  
- Improved handling of noisy, low-contrast, and partial fingerprint images  

---

## About the Project
This project estimates local ridge orientation and ridge frequency from fingerprint images using an adaptive CNN.

Why it is useful:
- Improves fingerprint image enhancement  
- Supports accurate minutiae extraction  
- Enhances reliability of biometric authentication systems  

General workflow:  
Fingerprint Image → Preprocessing (XSFFE and SNFFE) → Segmentation → Adaptive CNN → Orientation Map and Frequency Map → Post-processing → Final Output

---

## Dataset Used
👉 FFE Dataset

Dataset Details:
- Designed for fingerprint orientation, frequency, and foreground estimation  
- Contains both good and degraded quality fingerprint images  
- Suitable for evaluating robustness under noisy conditions  

---

## Dependencies Used
Python, TensorFlow or Keras, NumPy, OpenCV, Matplotlib, Scikit-learn

---

## EDA and Preprocessing
- Foreground segmentation to isolate fingerprint region  
- XSFFE for synthetic ridge enhancement and noise suppression  
- SNFFE for normalization of brightness, contrast, and ridge sharpness  
- Data augmentation such as rotation, flipping, and contrast adjustments  

---

## Model Training Info
- Architecture: Adaptive CNN  
- Convolution Layers: 6 with 3 by 3 kernels  
- Activation Function: ReLU  
- Optimizer: Adam  
- Batch Size: 32  
- Epochs: 100  
- Training Strategy: Pixel-wise dense supervision  

---

## Model Testing and Evaluation
- Evaluated using Mean Absolute Percentage Error (MAPE)  
- Tested on both good-quality and degraded fingerprint images  
- Performance compared with traditional CNN and regression-based methods  

---

## Results
- MAPE for good images: 4.58 percent  
- MAPE for degraded images: 5.02 percent  
- Average execution time: 0.012 seconds per fingerprint  
- Improved accuracy and robustness over existing methods  

---

## Limitations and Future Work
Limitations:
- Limited evaluation on latent fingerprints  
- Testing restricted to the FFE dataset  

Future Work:
- Integration into complete fingerprint recognition systems  
- Cross-sensor and cross-population evaluation  
- Handling extremely degraded or latent fingerprints  
- Detailed ablation studies for XSFFE and SNFFE  

---

## Deployment Info
- Suitable for real-time biometric systems  
- Can be integrated into automated fingerprint identification systems  
- Deployable as a backend service for fingerprint analysis  
