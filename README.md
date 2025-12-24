# 🩺 Breast Cancer Classification using Machine Learning

## 📌 Project Overview

This project focuses on **binary classification of breast cancer tumors** as **Malignant (M)** or **Benign (B)** using multiple machine learning models.
The goal is to compare different classifiers, tune their hyperparameters, and select the best-performing model based on recall, which is critical in medical diagnosis.

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

> Results may vary slightly depending on random seed and data split.

> **Recall was prioritized** to reduce false negatives in cancer detection.

---

## 📉 Visualizations
* Y-axis limited to `[0, 1]` for consistent comparison
  
The project generates:

- Recall comparison bar chart  
- Accuracy comparison bar chart  

Saved automatically in the `results/` directory:

- `recall_comparison.png`
- `accuracy_comparison.png`

---

⚙️ Techniques Used

*Pipelines for models requiring feature scaling
*RandomizedSearchCV for hyperparameter tuning
*Recall-based model selection
*Confusion Matrix, Accuracy, and Recall for evaluation
*Matplotlib visualizations for model comparison

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
├── results
```

---

## 🎯 Key Takeaways

* Random Forest and SVM achieved **perfect recall**
* Feature importance analysis confirmed medical relevance
* Recall-based optimization is crucial for healthcare ML problems

---

## 👨‍🎓 Author

**Madhav**
B.Tech Biotechnology | Aspiring Bioinformatics Professional

---


## 📜 License
This project is licensed under the MIT License.


---

