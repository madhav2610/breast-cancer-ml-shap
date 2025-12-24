
# 🩺 Breast Cancer Classification using Machine Learning

## 📌 Project Overview

This project focuses on the **classification of breast cancer tumors** as **Benign (B)** or **Malignant (M)** using multiple machine learning models.
The goal is to **compare different classifiers**, tune their hyperparameters, and select the **best-performing model based on recall**, which is critical in medical diagnosis.

---

## 🎯 Problem Statement

Early and accurate detection of breast cancer significantly improves patient outcomes.
In this project, machine learning models are trained on diagnostic features to predict whether a tumor is malignant or benign.

Since **false negatives (missing a malignant case)** are dangerous, **recall** is prioritized over accuracy.

---

## 📂 Dataset

* **Source:** Breast Cancer Wisconsin Dataset
* **Target Variable:** `diagnosis`

  * `B` → Benign (0)
  * `M` → Malignant (1)
* **Preprocessing Steps:**

  * Dropped unnecessary columns (`id`, `Unnamed: 32`)
  * Encoded target labels
  * Train-test split (80–20)

---

## 🧠 Models Implemented

The following models were trained and evaluated:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**
4. **AdaBoost Classifier**

---

## ⚙️ Techniques Used

* **Pipelines** for models requiring feature scaling
* **RandomizedSearchCV** for hyperparameter tuning
* **Recall-based model selection**
* **Confusion Matrix, Accuracy, and Recall** for evaluation
* **Matplotlib visualizations** for model comparison

---

## 📊 Evaluation Metrics

* **Recall Score (Primary Metric)**
* **Accuracy Score**
* **Confusion Matrix**

📌 *Recall is emphasized because failing to detect malignant tumors can have serious consequences.*

---

## 📈 Results Visualization

The project generates:

* Recall comparison bar chart
* Accuracy comparison bar chart

Saved automatically in the `results/` directory:

* `recall_comparison.png`
* `accuracy_comparison.png`

---

## 🏆 Best Model Selection

Models are compared based on **recall score**, and the model with the highest recall is considered the most suitable for this medical classification task i.e Random Forest.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib

---

## ▶️ How to Run

1. Clone the repository
2. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Run the script:

   ```bash
   python script.py
   ```

---

## 📌 Project Structure

```
Breast_Cancer_project/
│
├── data.csv
├── script.py
├── results/
│   ├── recall_comparison.png
│   └── accuracy_comparison.png
└── README.md
```

---

## 🔮 Future Work

* Add **model explainability techniques** (e.g., SHAP) to interpret feature importance
* Explore additional ensemble models
* Perform cross-dataset validation

---

## 👤 Author

**Madhav Takkar**
B.Tech Biotechnology
Machine Learning & Bioinformatics Enthusiast


