#  Hierarchical Animal Classifier

A deep learning web application that classifies animal images into **30 species** and **6 categories** using hierarchical classification with TensorFlow and Flask.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

##  Features

- **Hierarchical Classification**: Predicts both specific species and broad categories
- **30 Species Recognition**: Including mammals, birds, reptiles, amphibians, insects, and fish
- **6 Category Classification**: Mammifères, Oiseaux, Reptiles, Amphibiens, Insectes, Poissons
- **Interactive Web Interface**: Beautiful, responsive UI with drag-and-drop upload
- **Real-time Predictions**: Instant classification with confidence scores
- **Transfer Learning**: Built on MobileNetV2 pre-trained on ImageNet

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Athenwine/Pi-animal-classifier.git
cd animal-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install tensorflow==2.15.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
Pillow==10.0.0
flask==3.0.0
```

4. **Download the trained model**

Due to file size, the model is not included in the repository you can run the V2 file on colab to obtain it
Place the files in the project root:
- `hierarchical_animal_classifier.keras`
- `class_mappings.json`

5. **Run the application**
```bash
python hierarchical_app.py
```

6. **Open your browser**
```
http://127.0.0.1:5000
```

## 📦 Dataset

(https://drive.google.com/drive/folders/1diKq3WuyNhswZh_RqBjAOHxBgA5UiEx4?usp=sharing)

The model was trained on a custom dataset with the following structure:

```
dataset_bing/
├── Amphibiens/
│   ├── Grenouille/
│   └── Salamandre/
├── Insectes/
│   ├── Abeille/
│   ├── Coccinelle/
│   ├── Fourmi/
│   └── Papillon/
├── Mammiferes/
│   ├── Chat/
│   ├── Chien/
│   ├── Lion/
│   ├── Tigre/
│   ├── Éléphant/
│   └── ... (10 species)
├── Oiseaux/
│   ├── Aigle/
│   ├── Perroquet/
│   └── ... (6 species)
├── Poissons/
│   ├── Requin/
│   └── ... (4 species)
└── Reptiles/
    ├── Crocodile/
    └── ... (4 species)
```

**Total**: ~1,500 images across 30 species

## 🎓 Training Your Own Model

1. **Prepare your dataset** following the structure above

2. **Update the base directory** in `training/train_hierarchical.py`:
```python
BASE_DIR = "path/to/your/dataset_bing"
```

3. **Run training** (Google Colab recommended for GPU):
```bash
python training/train_hierarchical.py
```

Training time:
- With GPU (Google Colab): ~30-40 minutes
- Without GPU: ~2-3 hours

4. **Files generated**:
- `hierarchical_animal_classifier.keras` - Trained model
- `class_mappings.json` - Label mappings
- `training_history.png` - Performance curves
- `sample_training_images.png` - Dataset visualization

## 🔧 Configuration

### Modify Training Parameters

Edit `training/train_hierarchical.py`:

```python
IMG_SIZE = (224, 224)    # Image dimensions
BATCH_SIZE = 32          # Batch size
EPOCHS = 50              # Training epochs
```

### Change Web Server Port

Edit `hierarchical_app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

##  Troubleshooting

### Model won't load
- Ensure TensorFlow version matches: `pip install tensorflow==2.19.0`
- Verify model file exists and isn't corrupted
- Check `class_mappings.json` is in the same directory

### Out of memory during training
- Reduce batch size: `BATCH_SIZE = 16`
- Use Google Colab with GPU
- Close other applications

### Low accuracy
- Collect more training images (aim for 100+ per species)
- Balance dataset (similar images per class)
- Increase training epochs
- Check for mislabeled images

##  API Endpoints

### GET `/`
Returns the main web interface

### POST `/predict`
Classifies an uploaded image

**Request:**
```
Content-Type: multipart/form-data
file: <image_file>
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "species": "Chat",
      "species_confidence": 0.873,
      "species_confidence_percent": "87.3%",
      "category": "Mammiferes",
      "category_confidence": 0.952,
      "category_confidence_percent": "95.2%",
      "emoji": "🐱",
      "description": "Mammifère domestique, compagnon populaire"
    }
  ],
  "image": "data:image/jpeg;base64,..."
}
```

### GET `/info`
Returns model information

**Response:**
```json
{
  "total_species": 30,
  "total_categories": 6,
  "model_type": "Hierarchical Classification",
  "categories": ["Amphibiens", "Insectes", ...]
}
```

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  Acknowledgments

- **MobileNetV2**: Pre-trained model from TensorFlow/Keras
- **ImageNet**: Pre-training dataset
- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning framework

##  Contact

Your Name - [@Aws Ourari](https://www.linkedin.com/in/aws-ourari-2590891b6/) - awsourari123@gmail.com

Project Link: [https://github.com/Athenwine/Pi-animal-classifier](https://github.com/Athenwine/Pi-animal-classifier)


If you find this project useful, please consider giving it a star ⭐

---

**Made with ❤️ and 🐾**
