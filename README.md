# Stroke Prediction Project

## Overview
This project focuses on predicting the probability of stroke occurrence using medical and lifestyle features.  
The dataset used for this analysis consists of health indicators such as blood pressure, cholesterol levels, diabetes status, BMI, age, and kidney function metrics.  

The primary objective of this project is to build robust classification models that can:
1. Accurately classify stroke vs. no-stroke cases.
2. Produce well-calibrated probabilistic predictions for stroke risk assessment.

---

## Dataset
Two datasets were used:

- **P2_data_stroke_train.csv** – training and validation data used for model development.  
- **P2_data_stroke_test.csv** – unseen test data used for final predictions and submission.

The dataset is imbalanced, with significantly fewer stroke cases compared to non-stroke cases. Appropriate resampling methods (undersampling, oversampling, and hybrid techniques) were applied to mitigate this imbalance.

---

## Feature Engineering
A set of clinically relevant interaction and ratio features were created to enhance predictive power. Examples include:

- Blood pressure difference (`BP_Difference = Systolic - Diastolic`)  
- Cholesterol and lipid ratios (`LDL_HDL_Ratio`, `Cholesterol_Ratio`)  
- Interaction terms (`Age * BMI`, `Age * Systolic`)  
- Binary health condition flags (e.g., `BP_High`, `HDL_Low`, `A1C_High`, `eGFR_Low`)

These derived features capture key physiological relationships known to influence cardiovascular and stroke risk.

---

## Models Developed
Three supervised learning models were trained and evaluated:

| Model | Description | Key Techniques | Accuracy | Balanced Accuracy | KL Divergence (Log Loss) |
|--------|--------------|----------------|-----------|-------------------|--------------------------|
| **Logistic Regression** | Baseline linear model with regularization and undersampling | Feature selection with ANOVA F-test, robust scaling, `RandomUnderSampler` | 0.7050 | 0.7050 | 0.5930 |
| **Random Forest** | Ensemble model using bagged decision trees | Feature selection, balanced class weights, `RandomUnderSampler` | 0.7300 | 0.7280 | 0.5760 |
| **Support Vector Machine (SVM)** | Nonlinear classifier with RBF kernel | Robust scaling, feature selection, class rebalancing via undersampling | 0.7250 | 0.7240 | 0.5880 |

*Metrics are based on validation split performance.*

---

## Evaluation Metrics
The models were evaluated using the following metrics:

- **Accuracy**: Fraction of correctly classified samples.  
- **Balanced Accuracy**: Average of recall for both stroke and no-stroke classes, ensuring fairness under class imbalance.  
- **KL Divergence (Log Loss)**: Measures the divergence between predicted and true probability distributions; lower is better.

Balanced accuracy and KL divergence were chosen to evaluate both discrimination and calibration of probabilistic predictions.

---

## Final Predictions
Each model was used to generate probability predictions (`p(stroke=1|X)`) on the unseen **P2_data_stroke_test.csv** dataset.  

A final combined file (`stroke_predictions_final.csv`) was created with the following structure

---

## Tools and Libraries
The project was implemented in **Python 3.10+** using the following key libraries:

- **pandas**, **numpy** – data manipulation  
- **scikit-learn** – machine learning models, pipelines, and metrics  
- **imbalanced-learn (imblearn)** – resampling techniques for imbalance handling  

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Nishi-Pawar/Stroke_Prediction.git
   cd Stroke_Prediction
