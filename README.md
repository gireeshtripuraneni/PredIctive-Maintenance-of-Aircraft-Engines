# PredIctive-Maintenance-of-Aircraft-Engines
Hybrid Expert System + Machine Learning framework for Predictive Maintenance of Aircraft Engines. Includes RUL% prediction, expert rule-based alerts, engine condition classification (OK/WARNING/DANGER), hybrid decision logic, full notebook, dataset pipeline, and visualization.

# Hybrid Expert System and Machine Learning Model for Predictive Maintenance of Aircraft Engines ✈️🛠️

This project presents a complete hybrid Predictive Maintenance system for aircraft engines, combining **Machine Learning** with an **Expert System (rule-based AI)** to estimate engine health, detect failures early, and generate maintenance recommendations.  
The solution uses multivariate time-series sensor data to predict **Remaining Useful Life (RUL%)** and classify engine condition into **OK**, **WARNING**, or **DANGER**.

---

## 🚀 Key Features

### ✅ 1. Data Preprocessing & Feature Engineering
- Merge training, test, and ground-truth datasets  
- Remove constant-value and uninformative columns  
- Normalize sensor features  
- Compute **RUL%** using per-engine lifecycle normalization  
- Add time-based features such as short-window **sensor slopes**  

### ✅ 2. Machine Learning Models  
Multiple ML models were trained and benchmarked:
- Linear Regression  
- Huber Regression  
- Polynomial Regression  
- Random Forest  
- **XGBoost**  
- **LightGBM**  
- Support Vector Regression  
- Neural Network (Dense layers)

📌 **LightGBM & XGBoost achieved the best overall performance** with the lowest MAE & RMSE.

---

## 🧠 3. Expert System (Experta Rule Engine)

An interpretable, rule-based AI system was developed using `experta`, capturing domain-inspired behavior:

Rules include:
- 🔥 High vibration → HIGH alert  
- 🌡️ High temperature → MEDIUM alert  
- 📈 Rapid sensor slope → MEDIUM alert  
- ⚠️ Low RUL% → CRITICAL alert  
- 🤖 ML low RUL% + abnormal sensors → escalate to HIGH/DANGER  

The expert system outputs:
- `expert_alerts` (explainable messages)  
- `expert_severity` (OK, MEDIUM, HIGH, CRITICAL)  

---

## 🔄 4. Hybrid Decision Framework

Machine Learning + Expert System outputs are merged into a single decision:

| Condition | Description |
|----------|-------------|
| **MAINTENANCE NOW** | Expert severity HIGH/CRITICAL |
| **MAINTENANCE SOON** | ML model predicts RUL% ≤ threshold |
| **NO ACTION** | Engine is stable |

This hybrid system mirrors real-world aviation maintenance workflows, ensuring both safety and accuracy.

---

## 📊 5. Visualizations

The project includes:
- True vs Predicted RUL% distribution  
- Confusion matrix for engine condition  
- Engine time-series RUL% graph with expert alert markers  
- Final resampled data tables & CSV outputs  

---

## 📁 Files Included

- `Predictive_Maintenance_Experta.ipynb` — main notebook  
- Dataset files: `PM_train.xlsx`, `PM_test.xlsx`, `PM_truth.xlsx`  
- Output files:
  - `df_final_with_preds.xlsx`
  - `df_final_with_expert_hybrid.xlsx`
  - `engine_condition_distribution.png`
  - `top_10_critical_units.csv`
  - `Predictive_Maintenance_Project_Report.pdf`  

---

## ▶️ How to Run (Google Colab)

### **1. Open notebook in Colab**
Upload or open:

### **2. Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

!pip uninstall -y experta frozendict
!pip install git+https://github.com/niazangels/experta-py3.12.git
!pip install frozendict==2.3.8
