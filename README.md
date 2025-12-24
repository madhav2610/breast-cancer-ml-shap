

# 🩺 Breast Cancer Classification using Machine Learning & SHAP

## 📌 Project Overview

This project focuses on **binary classification of breast cancer tumors** as **Malignant (M)** or **Benign (B)** using multiple machine learning models.
The goal is to **maximize recall** (minimize false negatives), which is critical in medical diagnosis.

In addition to model comparison, **SHAP (SHapley Additive exPlanations)** is used to interpret the best-performing model and understand feature contributions.

---

## 🧠 Models Implemented

The following models were trained, tuned, and evaluated:

* **Logistic Regression** (with Standard Scaling + Hyperparameter Tuning)
* **Random Forest Classifier**
* **Support Vector Machine (SVM)** (with Standard Scaling)
* **AdaBoost Classifier**

Hyperparameter optimization was performed using **RandomizedSearchCV** with **Recall** as the primary scoring metric.

---

## 📊 Dataset

* **Dataset**: Breast Cancer Wisconsin Dataset
* **Target Variable**: `diagnosis`

  * `M` → Malignant (1)
  * `B` → Benign (0)
* **Preprocessing Steps**:

  * Dropped irrelevant columns: `id`, `Unnamed: 32`
  * Encoded target labels
  * Train-test split (80:20)

---

## ⚙️ Technologies & Libraries Used

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib
* SHAP

---

## 📈 Model Evaluation Metrics

Each model was evaluated using:

* **Confusion Matrix**
* **Recall Score** (Primary Metric)
* **Accuracy Score**

### 🔎 Model Performance Summary

| Model               | Recall | Accuracy |
| ------------------- | ------ | -------- |
| Logistic Regression | 0.95   | 0.93     |
| Random Forest       | 1.00   | 0.98     |
| SVM                 | 1.00   | 0.97     |
| AdaBoost            | 0.98   | 0.99     |

> **Recall was prioritized** to reduce false negatives in cancer detection.

---

## 📉 Visualizations

* Bar plots comparing **Recall** and **Accuracy** across models
* Y-axis limited to `[0, 1]` for consistent comparison

---

## 🔍 Model Explainability with SHAP

To interpret model predictions:

* The **best-performing Random Forest model** was selected
* **SHAP TreeExplainer** was used
* Generated:

  * SHAP Summary Plot
  * SHAP Feature Importance (Bar Plot)

### Why SHAP?

SHAP helps explain:

* How each feature contributes to predictions
* Direction of influence (positive or negative)
* Feature interactions

This improves **model transparency**, which is essential in healthcare applications.

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/breast-cancer-ml.git
cd breast-cancer-ml
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the script

```bash
python script.py
```

---

## 📌 Project Structure

```
├── data.csv
├── script.py
├── README.md
├── requirements.txt
```

---

## 🎯 Key Takeaways

* Random Forest and SVM achieved **perfect recall**
* Feature importance analysis confirmed medical relevance
* SHAP provided interpretable insights into model decisions
* Recall-based optimization is crucial for healthcare ML problems

---

## 👨‍🎓 Author

**Madhav Takkar**
B.Tech Biotechnology | Aspiring Bioinformatics Professional

---

## 📜 License

This project is for **educational and academic use**.

---


