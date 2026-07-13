# 📊 Breast Cancer Data Analysis App

An interactive web application built with **Streamlit** to explore, visualize, and analyze the **Breast Cancer Wisconsin (Diagnostic) Dataset**. This tool provides research-grade insights and allows users to perform custom data exploration.

---

## 🚀 Features

- **Dataset Overview**: Instantly view sample data, dataset dimensions, and key descriptive statistics.
- **Interactive Visualizations**:
  - **Correlation Heatmap**: Analyze relationship strengths between cellular features.
  - **Feature Distribution**: View KDE-smoothed histograms for individual features.
  - **Class Distribution**: Interactive pie chart displaying the ratio of Benign vs. Malignant cases.
- **Custom Data Upload**: Upload any custom CSV dataset to dynamically generate:
  - Data previews and shape summaries.
  - Comprehensive statistical summaries.
  - Missing value analysis.
  - Correlation heatmaps.

---

## 🛠️ Technology Stack

- **Frontend/Interactive UI**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
- **Data Visualizations**: [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
- **Dataset Source**: [Scikit-learn (load_breast_cancer)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---

## 📦 Installation & Setup

Follow these steps to run the application locally:

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install streamlit pandas matplotlib seaborn scikit-learn
```

### 3. Run the Application
Navigate to the project directory and run the Streamlit command:
```bash
streamlit run "AD 3-2/breastcancer.py"
```

Once running, the application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 📂 Project Directory Structure

```text
AD 3-2/
├── AD 3-2/
│   ├── breastcancer.py          # Main Streamlit App
│   ├── batch39 document.docx    # Project Report / Documentation
│   └── batch39 document.pdf     # PDF Report
└── README.md                    # Project Guide (This file)
```

---

## 👥 Authors
- **M. Bhargav** (2211CS010363)
- **P. Charan Sai** (2211CS010459)
- **N. Uday Kumar** (2211CS010407)
- **S. Harsha Vardhan** (2211CS010514)
