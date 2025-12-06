# ✈️ Flight Delay Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

This Machine Learning project aims to predict whether a commercial flight in the US will be **delayed (more than 15 minutes)** or on time, based on historical flight data.

The objective is to compare a linear approach (**Logistic Regression**) against a non-linear approach (**Decision Tree**) to identify risk factors and provide actionable insights for passengers and airlines.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Technologies Used](#-technologies-used)
4. [Installation & Execution](#-installation--execution)
5. [Project Structure](#-project-structure)
6. [Methodology](#-methodology)
7. [Key Results](#-key-results)

---

## 🧐 Project Overview
Flight delays cost billions of dollars annually and cause significant stress to travelers. This project uses supervised learning to address the following binary classification problem:
> **Will a specific flight be delayed at departure? (Yes/No)**

We implemented a complete pipeline ranging from **Exploratory Data Analysis (EDA)** to **Model Evaluation**, including advanced preprocessing techniques like **SMOTE** for class balancing.

## 📊 Dataset
* **Source:** [2015 Flight Delays and Cancellations (Kaggle)](https://www.kaggle.com/datasets/usdot/flight-delays)
* **Data Used:** A representative sample of **100,000 flights** was used to optimize computation time while maintaining statistical significance.
* **Target Variable:** `delayed`
    * `0`: On time or early.
    * `1`: Delayed (> 15 minutes).
* **Key Features:** Month, Day, Scheduled Departure Time (converted to Part of Day), Distance, Airline, Origin Airport.

## 🛠 Technologies Used
This project is built using **Python** with the following libraries:
* **Pandas & NumPy:** Data manipulation and cleaning.
* **Matplotlib & Seaborn:** Data visualization (EDA).
* **Scikit-Learn:** Modeling (LogisticRegression, DecisionTree), Pipelines, GridSearchCV, Metrics.
* **Imbalanced-learn:** Handling class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Joblib:** Efficient model saving and loading.

## 🚀 Installation & Execution

### 1. Prerequisites
Make sure to install the necessary dependencies via pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib kagglehub
```
### 2. Execution order
The notebooks must be executed sequentially, as each step generates processed files for the next one:

1_EDA.ipynb: Downloads data, explores statistics/correlations, and creates the cleaned file flight_delays_cleaned_sample.csv.

2_Preprocessing.ipynb: Performs cleaning, encodes categorical variables (OneHot), applies SMOTE to balance the dataset, and saves to processed_data.pkl.

3_Modeling.ipynb: Trains models, performs Hyperparameter Tuning (GridSearchCV) and saves the trained models.

4_Evaluation.ipynb: Loads test data and generates final comparison charts (ROC Curves, Confusion Matrices).

## 📂 Project Structure
├── 1_EDA.ipynb  
├── 2_Preprocessing.ipynb  
├── 3_Modeling.ipynb  
├── 4_Evaluation.ipynb  
├── flight_delays_cleaned_sample.csv  
├── processed_data.pkl  
├── model_lr.pkl  
├── model_dt.pkl  
└── README.md  

### 3. Méthodologie et Résultats
## 🧠 Methodology

### 1. Advanced Preprocessing & Feature Engineering
* **Time Binning:** Transformed raw hours (e.g., 1730) into categorical "Parts of the Day" (`Morning`, `Evening`, `Night`) to capture non-linear traffic patterns.
* **Cardinality Reduction:** Kept only the top 20 busiest airports to reduce noise and dimensionality.
* **Class Balancing (SMOTE):** Since real-world delays are the minority class (~20%), we used SMOTE to synthesize examples during training. This forces the model to learn delays effectively rather than ignoring them.

### 2. Modeling Strategy
We compared two distinct families of algorithms:
* **Baseline:** `DummyClassifier` (Majority Class) to set the minimum benchmark.
* **Linear Model:** `LogisticRegression` (Simple, interpretable).
* **Non-Linear Model:** `DecisionTreeClassifier` (Captures complex rules).
* **Optimization:** We used `GridSearchCV` to tune hyperparameters like `max_depth` and `min_samples_leaf` to prevent overfitting.

## 🏆 Key Results

The **Decision Tree** model significantly outperformed Logistic Regression, demonstrating that the relationships between delays and variables (weather, time, traffic) are complex and non-linear.

| Model | F1-Score | AUC (ROC) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | ~64% | ~0.65 | Struggles to separate classes linearly. |
| **Decision Tree** | **~76%** | **~0.75** | **Best Model.** Successfully captures complex patterns. |

### Main Influencing Factors (Feature Importance):
According to the Decision Tree analysis, delays are primarily driven by:
1.  **Departure Time** (`SCHEDULED_DEPARTURE`): Delays accumulate throughout the day ("snowball effect").
2.  **Seasonality** (`MONTH`): Strong correlation with Winter storms and Summer travel peaks.
3.  **Airline**: Structural differences between carriers (e.g., Low-cost vs Legacy) impact reliability.

---
*Project realized as part of the Introduction to Machine Learning module.*
