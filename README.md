# LoveLens 💕

## Overview

LoveLens is a machine learning-powered dashboard designed to analyze relationship compatibility and predict match outcomes using behavioral and personality-based features from online dating application data.

The project combines data analytics, machine learning, and interactive visualization through a modern Streamlit dashboard interface.

---

# Features

* Interactive Streamlit dashboard
* Relationship compatibility prediction
* Dataset exploration and analysis
* Machine learning workflow visualization
* Model performance evaluation
* Modern UI with custom styling and animations
* Responsive dashboard layout

---

# Dataset Information

The dataset contains behavioral and demographic features collected from online dating application interactions.

### Dataset Size

* **Rows:** 50,000
* **Columns:** 19

### Features Included

* Gender
* Sexual Orientation
* Location Type
* Income Bracket
* Education Level
* Interest Tags
* App Usage Time
* Swipe Right Ratio
* Likes Received
* Mutual Matches
* Bio Length
* Message Count
* Emoji Usage Rate
* Last Active Hour
* Match Outcome
* And more

### Target Variable

`match_outcome`

---

# Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* Plotly
* NumPy

---

# Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Dashboard Deployment

---

# Project Structure

```bash
LoveLens/
│
├── app.py
├── pages/
│   ├── predictor.py
│   ├── analytics.py
│   └── performance.py
│
├── data/
│   └── dating_app_behavior_dataset.csv
│
├── models/
│
├── assets/
│
└── README.md
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/jasmineyh/LoveLens.git
```

## Navigate to the Project Folder

```bash
cd LoveLens
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will launch automatically in your browser.

---

# Dashboard Pages

## Home

* Project overview
* Dataset statistics
* Workflow visualization
* Model summary

## Dataset Overview

* Dataset preview
* Feature information
* Data types
* Missing value analysis

## Prediction System

* User input prediction interface
* Match outcome prediction

## Analytics & Visualization

* Interactive charts
* Behavioral analysis
* Feature insights

## Model Evaluation

* Accuracy metrics
* Confusion matrix
* Performance visualization

---

# Future Improvements

* Real-time prediction API
* User authentication
* Advanced recommendation system
* Deep learning integration
* Cloud deployment
* Mobile responsive optimization

---

# Team Members

*Jasmine Chin Ying Hui | Lee Jian Cheng | Ng Yue Qhi | Nurul Afyqah binti Lukman | Ratu Ahdys Khairany | Tan Pei Shing
