<div id="top"></div>
<div align="center">

# 🚗 BMW Price Prediction
*Exploratory Data Analysis & Machine Learning on UK Used Car Data*

![last-commit](https://img.shields.io/github/last-commit/muhamadakmal1/bmw_price_prediction_eda_ml?style=flat&logo=git&logoColor=white&color=blue)
![repo-top-language](https://img.shields.io/github/languages/top/muhamadakmal1/bmw_price_prediction_eda_ml?style=flat&color=blue)
![repo-language-count](https://img.shields.io/github/languages/count/muhamadakmal1/bmw_price_prediction_eda_ml?style=flat&color=blue)

*Technologies: Python · Pandas · Scikit-learn · CatBoost · Seaborn · Matplotlib*

</div>

---

## 📚 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
  - [Data Cleaning](#data-cleaning)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Future Improvements](#future-improvements)

---

## 🔍 Overview
This project predicts the resale price of used BMW cars using the **100,000 UK Used Car Dataset** from Kaggle. It walks through the complete data-science lifecycle — from raw data exploration and cleaning, through insightful visualisations, all the way to training and comparing multiple regression models. The best model achieves **~96 % R²** accuracy on test data.

---

## 📦 Dataset

| Detail | Value |
|---|---|
| **Source** | [100,000 UK Used Car Dataset – Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) |
| **Subset used** | `bmw.csv` |
| **Rows** | ~10,000 |
| **Target** | `price` (£) |

| Column | Type | Description |
|---|---|---|
| `model` | Categorical | BMW model name (e.g. 3 Series, X5) |
| `year` | Integer | Registration year |
| `price` | Integer | Listed resale price in £ *(target)* |
| `transmission` | Categorical | Manual / Automatic / Semi-Auto |
| `mileage` | Integer | Odometer in miles |
| `fuelType` | Categorical | Petrol / Diesel / Hybrid / Electric |
| `tax` | Integer | Annual road tax in £ |
| `mpg` | Float | Miles per gallon |
| `engineSize` | Float | Engine size in litres |

---

## ✨ Features
- Full **Exploratory Data Analysis** with correlation heatmaps, distributions & trend lines
- **Outlier detection & removal** using box-plot analysis
- **Label Encoding & MinMax Scaling** for feature preparation
- Comparison of **3 regression models** — Linear Regression, Random Forest & CatBoost
- Clean evaluation with **R², MSE & MAE** metrics
- Actual vs Predicted price visualisations

---

## 🗂️ Project Structure
```
bmw_price_prediction_eda_ml/
│
└── bmw_price_prediction_eda_ml.ipynb   # Main notebook (EDA + ML pipeline)
```

---

## 📊 Workflow

### 🧹 Data Cleaning
- Loaded the BMW CSV and inspected shape, dtypes, and null counts
- Detected outliers per numeric column using **box-plots**
- Removed rows exceeding real-world thresholds (e.g. mileage > 200,000 miles, abnormal tax values)

### 📈 Exploratory Data Analysis
- **Correlation heatmap** — revealed strong negative links between `mileage`/`mpg` and `price`, and a positive link with `engineSize`
- **Histograms & KDE plots** — showed skew in price, mileage, and year distributions
- **Category counts** — broke down transmission types and fuel types
- **Trend lines** — mapped how price depreciates and tax changes over vehicle age

### ⚙️ Feature Engineering
- **Label Encoding** applied to `model`, `transmission`, and `fuelType`
- **MinMax Scaling** normalised all features to the [0, 1] range

### 🤖 Model Training & Evaluation
The data was split into train/test sets and three models were built and compared:

| Model | R² (Test) | Notes |
|---|---|---|
| Linear Regression | ~80 % | Baseline; misses non-linear patterns |
| Random Forest Regressor | ~95–96 % | Strong on feature interactions |
| **CatBoost Regressor** | **~96 %** | Best overall performer |

---

## 📉 Results
- **Mileage** is the strongest negative predictor — more miles driven = lower resale value
- **Engine size & year** are the strongest positive predictors
- **CatBoost** edges out the other models, confirming the relationship is substantially non-linear
- Actual vs Predicted scatter plots show tight alignment for the top models

---

## 🚀 Getting Started

### 📌 Prerequisites
- Python 3.7+
- Jupyter Notebook
- Basic understanding of Python & data science concepts

### 📥 Installation
```bash
# 1. Clone the repository
git clone https://github.com/muhamadakmal1/bmw_price_prediction_eda_ml.git
cd bmw_price_prediction_eda_ml

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn catboost jupyter

# 3. Place bmw.csv in the project root (download from the Kaggle link above)

# 4. Launch the notebook
jupyter notebook bmw_price_prediction_eda_ml.ipynb
```

---

## 🔮 Future Improvements
- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- Add XGBoost, LightGBM or a stacking ensemble
- Derive new features — *car age*, *price per mile*, etc.
- Replace Label Encoding with One-Hot Encoding
- Deploy the best model as a **Streamlit** or **Flask** web app

---

<div align="left"><a href="#top">⬆ Return to Top</a></div>
