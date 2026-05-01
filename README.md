#  FIFA Players Performance Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project that analyzes FIFA player data to predict market value and performance tiers.  
The project implements a robust preprocessing pipeline, advanced feature engineering, polynomial regression with regularisation, and multiple Naïve Bayes classifiers – all rigorously evaluated on a cleaned FIFA dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Objectives](#key-objectives)
- [Repository Structure](#repository-structure)
- [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
- [Models & Results](#models--results)
  - [1. Polynomial Regression](#1-polynomial-regression-task-4)
  - [2. Naïve Bayes Classification](#2-naïve-bayes-classification-task-6)
- [Running the Project](#running-the-project)
- [Results Summary](#results-summary)
- [Contributors](#contributors)

---

## Project Overview

This repository solves two main machine learning problems on the **FIFA Players Dataset**:

- **Regression** – predict the market value of a player (in `Value Per M$`).
- **Classification** – group players into four performance tiers (`Low`, `Mid`, `High`, `Elite`) based on their overall rating.

Both tasks share a unified preprocessing pipeline that includes:

- Handling of high-cardinality categorical features (1,009 teams, 164 countries)
- Outlier capping based on IQR
- Standard scaling and proper encoding (one‑hot + ordinal)
- Feature engineering via **continent mapping** and **K‑Means team‑tier clustering**

The models are evaluated using appropriate metrics and cross-validation techniques, following the structure of the **Machine Learning Assignment 2** from the course.

---

## Dataset

The dataset contains **19,667** FIFA player records (after cleaning). Each record includes:

| Column | Description |
|--------|-------------|
| Name | Player full name |
| Country | Nationality (164 unique values) |
| Position | Playing position (e.g. ST, GK, CB) |
| Age | Age in years |
| Overall_Rating | FIFA overall rating (0‑100) |
| Future Potential | Projected peak rating |
| Team | Current club (1,009 unique teams) |
| Value Per M$ | Market value in million USD (regression target) |
| Total_Stats Score | Composite in‑game stats aggregate |

> The original dataset is available at: [Google Drive link](https://drive.google.com/drive/folders/13Qb935zhUWRxn4yxMRqbsixXnEGLkhbC?usp=drive_link)

---

## Key Objectives

- **Exploratory Data Analysis (EDA)**: Understand distributions, correlations, and skewness of numerical features; identify outliers.
- **Preprocessing**: 80/20 train‑test split, encoding categorical columns, scaling numerical features, and outlier treatment using IQR capping.
- **Feature Engineering**:
  - Replace the 164 country values with **continent** (6 groups) to reduce noise.
  - Cluster 1,009 teams into **4 tiers** (`Elite`, `High`, `Medium`, `Low`) using K‑Means on average team rating, then encode ordinally to preserve rank.
- **Regression (Task 4)**: Build polynomial regression models with Ridge/Lasso regularisation to predict player value.
- **Classification (Task 6)**: Implement Gaussian, Bernoulli, and Complement Naïve Bayes classifiers for performance tier prediction, excluding `Overall_Rating` from features.

---

## Repository Structure

├── EDA_Preprocessing.ipynb # Full EDA, feature engineering, preprocessing pipeline, and data export

├── poly.ipynb # Polynomial regression (Task 4) with Ridge/Lasso

├── Naive-bias.ipynb # Naïve Bayes classification (Task 6) with scaling sensitivity analysis

├── X_train_transformed.csv # Preprocessed training features (output of EDA_Preprocessing)

├── X_test_transformed.csv # Preprocessed test features

├── y_train_reg.csv # Regression target (train)

├── y_test_reg.csv # Regression target (test)

├── y_train_clf.csv # Classification target (train)

├── y_test_clf.csv # Classification target (test)

├── Fifa_with_Continents.csv # Intermediate dataset after continent mapping

├── README.md # This file

└── requirements.txt # Python dependencies



---

## Feature Engineering & Preprocessing

All preprocessing steps are applied **only on the training set** and then propagated to the test set to prevent data leakage.

1. **Continent Mapping**  
   `Country` (164 categories) is replaced with `Continent` (Africa, Europe, Asia, North America, South America, Oceania), drastically reducing dimensionality while retaining geographical signals.

2. **Team‑Tier Clustering (K‑Means)**  
   Teams are grouped by their average `Overall_Rating`. K‑Means (k=4) segments them into four tiers: `Elite`, `High`, `Medium`, `Low`. The tier column is then ordinally encoded with the correct order, preserving the logical hierarchy.

3. **Outlier Handling**  
   Numerical features are clipped to the 1.5×IQR bounds computed from the training set. The same clipping is applied to the test set. The target `Value Per M$` is also capped to remove extreme outliers.

4. **ColumnTransformer Pipeline**  
   - **Numerical**: `SimpleImputer` (mean) → `StandardScaler`
   - **One‑Hot encoded**: `Continent`, `Position`
   - **Ordinal encoded**: `Team_Tier` (Low < Medium < High < Elite)

The transformed data is saved as `X_train_transformed.csv` / `X_test_transformed.csv` with clear feature names.

---

## Models & Results

### 1. Polynomial Regression (Task 4)

The regression target is `Value Per M$`. All models are trained on the preprocessed data.

- **Baseline Linear Regression**  
  Test R² = 0.8005, RMSE = 0.4986

- **Polynomial Degrees on Numerical Features**  
  Only the four numerical columns are expanded. The categorical features are kept as‑is.

  | Degree | Train R² | Test R² | Gap |
  |--------|----------|---------|-----|
  | 1 | 0.8147 | 0.8005 | 0.0142 |
  | 2 | 0.8611 | 0.8498 | 0.0113 |
  | 3 | 0.9063 | 0.8933 | 0.0130 |
  | **4** | **0.9315** | **0.9158** | 0.0157 |

  **Degree 4** was selected as it gives the highest test R² with a very small increase in overfitting.

- **Regularisation (Ridge & Lasso) on Degree 4**

  | Model | Best α | Test RMSE |
  |-------|--------|-----------|
  | Ridge | 12.7427 | **0.3204** |
  | Lasso | 0.0010 | 0.3256 |

  Ridge marginally outperforms Lasso because the polynomial + one‑hot features contain many correlated variables – Ridge spreads the penalty smoothly, while Lasso unnecessarily zeroes out some informative interactions.

- **Lasso Feature Selection**  
  Lasso with α=0.001 zeroes out 32 of 92 coefficients, discarding certain higher‑order terms like `Age × Future Potential` and `Future Potential²`. This confirms that not all interactions are equally important, but retaining them all (as Ridge does) yields a slight edge in predictive accuracy.

> The best model reduces the RMSE by **35%** compared to the baseline linear regression.

### 2. Naïve Bayes Classification (Task 6)

The classification target is `Performance_Tier` (Low, Mid, High, Elite). `Overall_Rating` is excluded because it directly defines the tiers.

Three Naïve Bayes variants were trained on the preprocessed data:

| Model | Accuracy | Key Behaviour |
|-------|----------|---------------|
| GaussianNB | **0.792** | Works well on roughly Gaussian numerical features |
| ComplementNB | 0.566 | Moderate performance, struggles with Elite class |
| BernoulliNB | 0.520 | Only predicts the two majority classes |

**GaussianNB** is the most appropriate because the numerical features follow approximately normal distributions, matching its core assumption. BernoulliNB expects binary features and fails to predict minority classes.

- **Scaling Sensitivity**  
  GaussianNB produces identical results with and without `StandardScaler`. Since it estimates per‑class means and variances, scaling changes the units but not the probability ratios – making it entirely unaffected by scaling.

---

## Running the Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/Dov-elhacker/fifa-players-analysis.git
   cd fifa-players-analysis

   
