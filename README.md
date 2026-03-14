# 🎓 Student Performance Prediction

An AI-powered web application that predicts a student's **CGPA (Cumulative Grade Point Average)** on a 10.0 scale using machine learning. The system is built with an ensemble of trained ML models and wrapped in an interactive **Streamlit** web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Input Features](#input-features)
- [Contributing](#contributing)

---

## 🔍 Overview

This project addresses the challenge of early identification of students at academic risk by predicting their CGPA based on a comprehensive set of demographic, behavioral, and academic factors. By leveraging multiple machine learning algorithms—including ensemble and stacking methods—the system achieves a test R² of **0.909**, meaning it explains over 90% of the variance in student CGPA.

The interactive Streamlit app lets educators, students, or researchers:
- Select from 10 trained ML models
- Enter student profile information
- Receive an instant CGPA prediction with a performance category
- View model statistics and feature importance charts

---

## 🚀 Live Demo

Run locally (see [Installation](#installation)) or deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting your GitHub repository and pointing the main file to `app.py`.

---

## ✨ Features

- **Multi-model selection** — choose from 10 trained models via the sidebar
- **Real-time CGPA prediction** on a 10.0 scale
- **Performance categories**: Outstanding 🌟, Excellent ⭐, Very Good ✨, Good 👍, Average 📚, Needs Improvement 💪
- **Model confidence metrics** — R², RMSE, and MAE displayed for the selected model
- **Feature importance visualization** using interactive Plotly charts
- **Responsive, animated UI** with gradient CSS styling
- **Handles missing inputs** gracefully using median/mode imputation

---

## 🛠️ Tech Stack

| Category | Libraries / Tools |
|---|---|
| **Web Framework** | [Streamlit](https://streamlit.io/) |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly (Express & Graph Objects), Matplotlib, Seaborn |
| **Model Persistence** | Pickle |
| **Notebook** | Jupyter Notebook |
| **Language** | Python 3.x |

---

## 📊 Dataset

| Property | Value |
|---|---|
| **File** | `Students_Performance_data_set.csv` / `.xlsx` |
| **Records** | 1,194 students |
| **Input Features** | 31 |
| **Target Variable** | Current CGPA (0–10 scale) |

### Data Preprocessing Steps

1. **Missing value imputation** — numeric columns filled with median; categorical columns filled with mode
2. **Duplicate removal**
3. **Categorical encoding** — `LabelEncoder` applied to all categorical features
4. **Scale conversion** — CGPA originally on a 4.0 scale; multiplied by 2.5 to convert to 10.0
5. **Feature scaling** — `StandardScaler` applied for linear models; raw features used for tree-based models

---

## 🤖 Machine Learning Models

The training pipeline (see `model1.py` and `model.ipynb`) trains and evaluates **9 models**:

### Base Models
| Model | Description |
|---|---|
| Linear Regression | Baseline linear model (scaled features) |
| Ridge Regression | L2-regularized linear model (scaled features) |
| Random Forest | 100-tree ensemble of decision trees |
| XGBoost | Extreme Gradient Boosting |
| LightGBM | Fast, efficient gradient boosting |

### Ensemble Models
| Model | Description |
|---|---|
| Voting Ensemble | Simple average of base model predictions |
| Weighted Voting Ensemble | Weighted average (higher weight for better models) |
| **Stacking Ensemble** ✅ | Meta-learner (Ridge) trained on base model predictions — **best overall** |
| Stacking-XGBoost | Meta-learner (XGBoost) trained on base model predictions |

All trained models are saved to the `models/` directory and loaded at app startup.

---

## 📁 Project Structure

```
Student-Performance-Prediction/
│
├── app.py                              # Streamlit web application
├── model1.py                           # Model training script
├── model.ipynb                         # Jupyter Notebook (step-by-step pipeline)
│
├── Students_Performance_data_set.csv   # Main dataset (CSV)
├── Students_Performance_data_set.xlsx  # Main dataset (Excel)
│
└── models/                             # Saved model artifacts
    ├── all_models.pkl                  # All trained models
    ├── best_model.pkl                  # Best model (Stacking Ensemble)
    ├── scaler.pkl                      # Fitted StandardScaler
    ├── feature_names.pkl               # Feature column names
    ├── important_features.pkl          # Top-10 important features
    ├── label_encoders.pkl              # Fitted LabelEncoders
    ├── median_values.pkl               # Median values for imputation
    └── model_results.csv               # Performance metrics for all models
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/SHUBHDEEP11103/Student-Performance-Prediction.git
cd Student-Performance-Prediction

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** See `requirements.txt` for the full list of pinned dependencies.

---

## ▶️ Usage

### Option 1 — Run the Web App (Recommended)

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

**Workflow in the app:**
1. Select a model from the sidebar (default: Stacking Ensemble)
2. Fill in the student's details in the input form
3. Click **Predict CGPA**
4. View the predicted CGPA, performance category, and model metrics

### Option 2 — Retrain the Models

```bash
python model1.py
```

This retrains all models on the dataset, evaluates them, and saves updated artifacts to the `models/` directory.

### Option 3 — Explore the Notebook

```bash
jupyter notebook model.ipynb
```

The notebook walks through the complete ML pipeline step-by-step with explanations, visualizations, and evaluation metrics.

---

## 📈 Model Performance

Performance on the held-out test set (20% of 1,194 records):

| Model | Test R² | Test RMSE |
|---|---|---|
| **Stacking Ensemble** 🏆 | **0.909** | **0.539** |
| Weighted Voting Ensemble | 0.903 | 0.558 |
| Voting Ensemble | 0.901 | 0.564 |
| Random Forest | 0.893 | 0.587 |
| XGBoost | 0.894 | 0.585 |
| LightGBM | 0.877 | 0.629 |
| Ridge Regression | 0.789 | 0.821 |
| Linear Regression | 0.790 | 0.822 |

> All RMSE values are on the 10.0-point CGPA scale.

---

## 🗂️ Input Features

The model uses **31 features** collected from student surveys:

### Demographic
| Feature | Description |
|---|---|
| University Admission Year | Year the student was admitted |
| Gender | Male / Female |
| Age | Student's age |
| H.S.C Passing Year | Year of higher secondary completion |
| Program | Enrolled degree program |
| Current Semester | Current semester number |

### Study Behavior
| Feature | Description |
|---|---|
| Daily Study Hours | Hours spent studying each day |
| Daily Study Sessions | Number of study sessions per day |
| Preferred Learning Mode | Online or Offline |
| Average Class Attendance | Attendance percentage |

### Resources & Infrastructure
| Feature | Description |
|---|---|
| University Transportation | Uses university transport (Yes/No) |
| Smartphone Usage | Has a smartphone (Yes/No) |
| Personal Computer | Has a personal computer (Yes/No) |
| Living Situation | With whom the student lives |

### Academic Engagement
| Feature | Description |
|---|---|
| Merit Scholarship | Has a meritorious scholarship (Yes/No) |
| Academic Probation | Has ever been on probation (Yes/No) |
| Suspension | Has ever been suspended (Yes/No) |
| Teacher Consultancy | Consults teachers for academic problems (Yes/No) |
| Co-curricular Activities | Participates in co-curricular activities (Yes/No) |

### Skills & Interests
| Feature | Description |
|---|---|
| Skills | Current technical/soft skills |
| Daily Skill Development Hours | Hours spent on skill development |
| Area of Interest | Student's domain of interest |

### Well-being
| Feature | Description |
|---|---|
| Daily Social Media Hours | Hours spent on social media per day |
| English Proficiency | Basic / Intermediate / Advanced |
| Relationship Status | Single / In a relationship / etc. |
| Health Issues | Any ongoing health issues (Yes/No) |
| Physical Disabilities | Any physical disability (Yes/No) |

### Academic Performance
| Feature | Description |
|---|---|
| **Previous SGPA** ⭐ | SGPA from the most recent semester (most important feature) |
| **Credits Completed** ⭐ | Total academic credits completed to date |
| Monthly Family Income | Family's monthly income bracket |

> ⭐ Top predictors identified by Random Forest feature importance.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👥 Authors

- **Shubham Kumar Jha** — Model development & training pipeline (TEAM SAS)
- **adamya1231** — Streamlit web application

---

*Built with ❤️ using Python, scikit-learn, XGBoost, LightGBM, and Streamlit.*
