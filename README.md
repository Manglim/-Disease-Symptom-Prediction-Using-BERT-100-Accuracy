# Disease Prediction Using BERT

## Overview
This project uses BERT (`bert-base-uncased`) to predict diseases from symptom descriptions in the Disease Symptom Description Dataset. It preprocesses symptom data, trains a BERT model, evaluates performance with visualizations, and predicts diseases from user-input symptoms, achieving 100% accuracy on the test set.

## Functionality
1. **Preprocessing**:
   - Loads `dataset.csv`, removes invalid rows, and combines symptoms into `symptoms_text`.
   - Encodes diseases with `LabelEncoder`.

2. **Training**:
   - Fine-tunes BERT for 3 epochs (batch size=16, lr=5e-5).

3. **Evaluation**:
   - Reports precision, recall, F1-score, and accuracy per epoch.
   - Final classification report for 41 diseases.

4. **Visualization**:
   - Training loss curve, confusion matrix, and evaluation metrics bar plot.

5. **Prediction**:
   - Predicts disease from space-separated symptom input.

## Frameworks and Libraries
- **Python**: Core language.
- **Pandas**: Data handling (`pd`).
- **NumPy**: Numerical ops (`np`).
- **PyTorch**: Deep learning (`torch`).
- **Transformers**: BERT (`transformers`).
- **Scikit-learn**: Utilities (`train_test_split`, `LabelEncoder`, metrics).
- **Matplotlib**: Plots (`plt`).
- **Seaborn**: Visualization (`sns`).

## Dataset
- Source: `dataset.csv` from `/kaggle/input/disease-symptom-description-dataset/`.
- Rows: 4,920 after preprocessing.
- Columns: `Disease` (41 unique), `Symptom_1` to `Symptom_17` combined into `symptoms_text`.

## Key Features
- **BERT-Powered**: Contextual symptom analysis.
- **Perfect Accuracy**: 1.00 across all metrics.
- **Visual Insights**: Loss, confusion matrix, and metrics plots.
- **Interactive**: Real-time disease prediction.

## Installation
```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn
