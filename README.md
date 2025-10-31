# Semantic Segmentation on Oxford-IIIT Pet Dataset

This project implements a **semantic segmentation model** using a **U-Net-style architecture** with **VGG16** as the encoder.  
The goal is to classify each pixel in an image as *pet*, *background*, or *border region*.

---

## 🎯 Objective
Build an encoder-decoder network that performs pixel-wise classification of pet images.  
This project demonstrates how transfer learning and skip connections can be combined for high-quality segmentation.

---

## 🧩 Methodology

| Step | Description |
|------|--------------|
| **1. Data Loading** | Loaded the Oxford-IIIT Pet dataset (images + segmentation masks). |
| **2. Preprocessing** | Resized to 128 × 128, normalized pixel values, prepared batches. |
| **3. Encoder** | Used pretrained **VGG16** with frozen weights as feature extractor. |
| **4. Decoder** | Built custom up-sampling path with `Conv2DTranspose`, `BatchNorm`, and skip connections. |
| **5. Compilation** | Loss – `SparseCategoricalCrossentropy(from_logits=True)`; Metric – `accuracy`. |
| **6. Training** | Trained for multiple epochs to predict masks for each pixel class. |

---

## 🧠 Model Architecture

- **Encoder:** VGG16 convolutional blocks (`block1_pool` → `block5_pool`)  
- **Decoder:** Series of transposed convolutions + skip connections  
- **Output:** Pixel-wise logits for three classes (background, pet, border)

---

## 📊 Results

| Metric | Value |
|---------|-------|
| Accuracy | ≈ 95 % (on validation subset) |
| Loss | steadily decreased with epochs |

- The model successfully learned clear boundaries around pets.
- Outputs visually align with ground-truth masks.

| Input | Ground Truth | Prediction |
|--------|--------------|-------------|
| ![Pet](images/pet_mask_example.png) | ![Mask](images/prediction_overlay.png) | ✅ Accurate segmentation |

---

## ⚙️ Requirements

```bash
pip install tensorflow keras numpy matplotlib opencv-python
