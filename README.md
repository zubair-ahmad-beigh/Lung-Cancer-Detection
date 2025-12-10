# 🫁 Lung Cancer Detection using CNN

A deep learning-based application for detecting lung cancer from medical images using Convolutional Neural Networks (CNN) with TensorFlow.

## 📋 Overview

Yeh project lung cancer detection ke liye ek complete solution hai jo:
- Medical images (DICOM aur regular image formats) ko process karta hai
- 3D CNN model use karta hai for accurate detection
- User-friendly GUI provide karta hai
- Training aur prediction dono features support karta hai

## ✨ Features

- **Data Import**: DICOM files aur regular images (PNG, JPG, JPEG, BMP, TIFF) ko import kar sakte hain
- **Data Preprocessing**: Automatic image preprocessing aur normalization
- **Model Training**: 3D Convolutional Neural Network training
- **Image Prediction**: Trained model se single image prediction
- **GUI Interface**: Easy-to-use Tkinter-based graphical interface

## 🛠️ Requirements

Project ko run karne ke liye yeh dependencies chahiye:

```
numpy>=1.19.0
pandas>=1.3.0
pydicom>=2.2.0
matplotlib>=3.3.0
opencv-python>=4.5.0
tensorflow>=2.8.0
scikit-learn>=0.24.0
Pillow>=8.0.0
```

## 📦 Installation

1. **Repository clone karein:**
```bash
git clone https://github.com/zubair-ahmad-beigh/Lung-Cancer-Detection.git
cd Lung-Cancer-Detection
```

2. **Dependencies install karein:**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### GUI Application (Recommended)

Main application ko run karne ke liye:

```bash
python lcd_cnn.py
```

**Steps:**
1. **Import Data**: "Import Data" button click karein aur images wala folder select karein
   - Folder structure: `aca/`, `normal/`, `scc/` (ya similar structure)
   - Ya directly images folder mein ho sakti hain

2. **Pre-Process Data**: Data ko preprocess karne ke liye button click karein
   - Yeh step `imageDataNew-10-10-5.npy` file generate karega

3. **Train Data**: Model ko train karne ke liye button click karein
   - Trained model `lung_cancer_model.ckpt` files mein save hoga

4. **Predict Image**: Single image predict karne ke liye button click karein
   - Image select karein aur result dekhein

### Headless Training (Command Line)

Agar GUI ke bina train karna hai:

```bash
python train_headless.py
```

## 📁 Project Structure

```
Lung-Cancer-Detection/
├── lcd_cnn.py              # Main GUI application
├── train_headless.py       # Headless training script
├── requirements.txt        # Python dependencies
├── stage1_labels.csv       # Labels file (optional)
├── Images/                 # Application images
│   └── Lung-Cancer-Detection.jpg
├── sample_images/          # Sample dataset
│   ├── train/
│   │   ├── aca/           # Adenocarcinoma images
│   │   ├── normal/        # Normal lung images
│   │   └── scc/           # Squamous cell carcinoma images
│   └── test/
│       ├── aca/
│       ├── normal/
│       └── scc/
└── README.md
```

## 🧠 Model Architecture

Yeh project 3D CNN architecture use karta hai:
- **Input**: 10x10x5 image slices
- **Layers**: 5 convolutional layers with max pooling
- **Filters**: 32 → 64 → 128 → 256 → 512
- **Output**: Binary classification (Cancer/No Cancer)

## 📊 Supported Image Formats

- **DICOM**: `.dcm`, `.dicom` (Medical imaging format)
- **Regular Images**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

## ⚙️ Configuration

Default settings:
- Image size: 10x10 pixels
- Number of slices: 5
- Keep rate (dropout): 0.8
- Learning rate: 1e-3
- Epochs: 5 (headless training)

## 📝 Notes

- Agar preprocessed data already hai (`imageDataNew-10-10-5.npy`), to training directly start ho sakti hai
- Agar trained model already hai (`lung_cancer_model.ckpt` files), to prediction directly use ho sakti hai
- Labels CSV file optional hai - agar nahi hai to folder names se automatically infer hoga

## 🤝 Contributing

Contributions welcome hain! Agar aap improvements chahte hain:
1. Fork karein repository
2. Changes karein
3. Pull request submit karein

## 📄 License

Yeh project open source hai. Free use karein!

## 👤 Author

**Zubair Ahmad Beigh**
- GitHub: [@zubair-ahmad-beigh](https://github.com/zubair-ahmad-beigh)

## 🙏 Acknowledgments

- TensorFlow team for deep learning framework
- Medical imaging community for datasets and research

---

**Note**: Yeh tool educational purposes ke liye hai. Medical diagnosis ke liye always qualified healthcare professionals se consult karein.

