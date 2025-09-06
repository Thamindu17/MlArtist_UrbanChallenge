# MlArtist Urban Challenge 🏙️

A deep learning project for automated urban issue classification using Convolutional Neural Networks (CNN). This system can identify and classify various urban problems from images, making it valuable for smart city applications and urban management.

## 🎯 Project Overview

This project tackles the challenge of automated urban issue detection by training a CNN model to classify different types of urban problems. The system can identify six main categories of urban issues from photographs, enabling automated monitoring and faster response times for city management.

## 🎬 Demo

📺 **Project Demo Video**: [https://youtu.be/gS_7ICu2u2g](https://youtu.be/gS_7ICu2u2g)

## 📊 Dataset

The dataset contains **14,457 total images** split across:
- **Training**: 11,564 images
- **Validation**: 1,444 images  
- **Testing**: 1,446 images

### 🏷️ Classes

The model classifies urban issues into **6 categories**:

| Class | Description | Training Samples |
|-------|-------------|------------------|
| `Infrastructure_Damage_Concrete` | Damaged concrete structures | 1,928 |
| `Domestic_trash` | Domestic waste and litter | 1,928 |
| `Vandalism_Graffiti` | Graffiti and vandalism | 1,927 |
| `Road_Issues_Pothole` | Potholes in roads | 1,927 |
| `Road_Issues_Damaged_Sign` | Damaged road signs | 1,927 |
| `Parking_Issues_Illegal_Parking` | Illegal parking violations | 1,927 |

The dataset is well-balanced with approximately **1,927-1,928 samples per class**.

## 🏗️ Model Architecture

The project implements a **Convolutional Neural Network (CNN)** optimized for multi-class image classification:

- **Architecture**: Custom CNN with multiple convolutional layers
- **Input**: RGB images of urban scenes
- **Output**: 6-class classification probabilities
- **Framework**: Python with deep learning libraries

### 📈 Architecture Diagrams

The repository includes detailed architecture diagrams:
- `Architecture Diagram/CNN Model Architecture.png` - CNN model structure
- `Architecture Diagram/Preprocessing & Training Pipeline Diagram.png` - Data pipeline

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or Google Colab
- Required libraries (see installation section)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Thamindu17/MlArtist_UrbanChallenge.git
cd MlArtist_UrbanChallenge
```

2. **Install dependencies**:
The project uses Google Colab for training. The main notebook (`MLArtists_FinalNotebook.ipynb`) includes installation commands for:
- Roboflow (for dataset management)
- TensorFlow/Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

3. **Download datasets**:
The notebook automatically downloads and processes datasets from Roboflow workspaces.

### Usage

#### 🎓 Training

1. Open `MLArtists_FinalNotebook.ipynb` in Google Colab
2. Run the setup and installation cells
3. Execute the data download and preprocessing sections
4. Train the CNN model using the provided pipeline
5. The trained model will be saved as `MlArtist_FinalModel.pkl`

#### 🔮 Inference

Use the trained model (`MlArtist_FinalModel.pkl`) to classify new urban images:

```python
# Load the trained model
import pickle
with open('MlArtist_FinalModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Classify new image
prediction = model.predict(preprocessed_image)
```

## 📁 Project Structure

```
MlArtist_UrbanChallenge/
├── MLArtists_FinalNotebook.ipynb    # Main training notebook
├── MlArtist_FinalModel.pkl          # Trained model
├── Dataset/                         # Dataset files
│   ├── train.csv                   # Training data paths and labels
│   ├── validate.csv                # Validation data paths and labels
│   └── test.csv                    # Test data paths and labels
├── Architecture Diagram/           # Model architecture diagrams
│   ├── CNN Model Architecture.png
│   └── Preprocessing & Training Pipeline Diagram.png
├── testing notebooks/              # Experimental notebooks
│   ├── baseline_training.ipynb
│   ├── database_build.ipynb
│   ├── hier_training.ipynb
│   ├── multilabel_stratified_split.ipynb
│   └── test.ipynb
├── MLArtist.ipynb - Colab.pdf      # Colab notebook PDF
├── MLArtist_report.pdf             # Detailed project report
├── youtube link-.txt               # Demo video link
└── README.md                       # This file
```

## 📋 Results

- **Balanced Dataset**: ~1,927 samples per class ensuring fair training
- **Multi-class Classification**: Successfully classifies 6 types of urban issues
- **Real-world Application**: Demonstrated effectiveness on urban imagery
- **Scalable Solution**: Pipeline can be extended for additional urban issue types

Detailed results and performance metrics are available in `MLArtist_report.pdf`.

## 🔬 Experimental Notebooks

The `testing notebooks/` directory contains various experimental approaches:

- `baseline_training.ipynb` - Baseline model experiments
- `database_build.ipynb` - Dataset construction pipeline
- `hier_training.ipynb` - Hierarchical classification approach
- `multilabel_stratified_split.ipynb` - Advanced data splitting strategies
- `test.ipynb` - Model testing and evaluation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is available for educational and research purposes. Please check with the authors for commercial use.

## 👥 Authors

- **Thamindu17** - *Project Lead* - [GitHub Profile](https://github.com/Thamindu17)

## 🙏 Acknowledgments

- Roboflow for dataset hosting and management
- Google Colab for providing free GPU resources
- Contributors to the urban imagery datasets
- Open source community for the machine learning libraries

## 📞 Contact

For questions or collaborations, please reach out through:
- GitHub Issues: [Project Issues](https://github.com/Thamindu17/MlArtist_UrbanChallenge/issues)
- Demo Video: [YouTube](https://youtu.be/gS_7ICu2u2g)

---

**Made with ❤️ for Smart Cities and Urban Management**