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
.
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
