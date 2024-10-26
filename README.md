# 🧠 Brain Tumor Semantic Segmentation

This project implements a semantic segmentation model designed to detect and segment brain tumors from CT scan images using the U-Net architecture. The objective is to identify tumor regions in medical images accurately, providing critical information for diagnosis and treatment planning. This repository contains all the necessary files to train, evaluate, and deploy the model, including a Streamlit application for an interactive user experience.

---
![Screenshot 2024-10-26 005550](https://github.com/user-attachments/assets/e5c9b536-2d7a-4adc-a689-b1df041d894a)


## 📊 Model Architecture

The model leverages the U-Net architecture, a popular convolutional neural network architecture for semantic segmentation. TensorFlow is used for building and training the model. U-Net's encoder-decoder structure enables it to capture high-level contextual information and retain spatial information effectively, making it highly suitable for medical image segmentation tasks.

### Metrics

The model is evaluated using the following performance metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **Intersection over Union (IoU)**

All metric calculations are implemented within `utils.py` and can be reviewed in the `Brain_Tumor_Segmentation.ipynb` notebook.

---

## 📈 Results

The evaluation results are documented in `Brain_Tumor_Segmentation.ipynb`. These results showcase the model's accuracy in segmenting brain tumor regions, along with detailed performance statistics for each metric.

---

## 📂 Dataset
The Tumor Segmentation Dataset is designed specifically for the TumorSeg Computer Vision Project, which focuses on Semantic Segmentation. The project aims to identify tumor regions accurately within Medical Images using advanced techniques.
The dataset includes 2146 images.
Tumors are annotated in COCO Segmentation format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* No image augmentation techniques were applied.
---

## 🧰 Utilities

The `utils.py` file includes functions for:
- Custom metrics calculation (Accuracy, Precision, Recall, IoU)
- Visualization utilities to display segmentation masks and model outputs

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue if you have suggestions, questions, or improvements.

