# KNN Image Classification Project - Team A

## Overview

An introductory AI project implementing K-Nearest Neighbors (KNN) classification for real-world vehicle and pedestrian detection. This project focuses on understanding overfitting, regularization techniques, and fundamental machine learning mathematics through hands-on image classification.

## Learning Objectives

- **Understand overfitting** and how to fix it using penalty regularization
- **Learn simple graph visualization** for performance analysis
- **Cover basic mathematics of machine learning** through practical implementation

## Project Requirements

### Dataset Classes
The project classifies the following real-world objects:
- Motorcycles
- Cars  
- Trucks
- Pedestrians
- Cyclists

All images should be compressed to **32x32 pixels** for consistency.

## Implementation Steps

1. **Data Collection**: Find clear images of the target classes in real-world settings
2. **Data Preparation**: Label pictures and split into training and test datasets
   - Constraint: `|training data| <= |test data|`
3. **Data Processing**: Implement dataset and image processing pipeline
4. **KNN Implementation**: Build custom KNN classifier and test performance
5. **Performance Analysis**: Plot performance vs. model complexity for both training and test datasets (I suggest using AI for the plots, but doing everything elsemanually)
6. **Optimization**: Implement penalty regularization to prevent overfitting and repeat analysis

## Must Have Features

- ✅ **Custom KNN implementation** (no external KNN libraries)
- ✅ **Overfitting analysis** with visual bias-variance tradeoff plots
- ✅ **Decision classifier** achieving >80% validation accuracy
- ✅ **Comprehensive technical documentation**

## Nice to Have Features

- 🎯 **Cross evaluation** on test vs training data
- 🎯 **Hyperparameter optimization** using grid search

## Dependencies

```
python3
PIL (Python Imaging Library)
sklearn (scikit-learn)
matplotlib
numpy
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd knn-image-classification
```

2. Install required dependencies:
```bash
pip install pillow scikit-learn matplotlib numpy
```

## Usage

1. **Prepare your dataset**: Place images in appropriate class folders under `data/`
2. **Run the preprocessing script**: `python preprocess_data.py`
3. **Train the KNN model**: `python train_knn.py`
4. **Evaluate performance**: `python evaluate_model.py`
5. **Generate plots**: `python plot_performance.py`

## Project Structure

```
knn-image-classification/
├── README.md
├── requirements.txt
├── data/
│   ├── motorcycles/
│   ├── cars/
│   ├── trucks/
│   ├── pedestrians/
│   └── cyclists/
├── src/
│   ├── knn_classifier.py
│   ├── data_preprocessing.py
│   ├── performance_analysis.py
│   └── regularization.py
├── plots/
├── docs/
└── tests/
```

## Performance Metrics

The project aims to achieve:
- **>80% validation accuracy** on the test dataset
- Clear demonstration of overfitting mitigation through regularization
- Visual evidence of bias-variance tradeoff optimization

## Technical Documentation

Detailed technical documentation including:
- Mathematical foundations of KNN
- Overfitting analysis methodology
- Regularization techniques implementation
- Performance evaluation metrics

## AI Project Assessment

**STRONGLY APPROVED** - This image-based modification transforms the introductory project from abstract ML concepts to directly applicable automotive computer vision, providing superior preparation for the main CNN-based ethical decision system while maintaining all original learning objectives around overfitting, regularization, and ML mathematics.

## Contributing

1. Clone the repo 
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Use Slack

## Acknowledgments

- Thomas Mbrice
- Ammar Ali
- Dataset contributors
