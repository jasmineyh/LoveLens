import streamlit as st
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="LoveLens Dashboard",
    page_icon="💕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/dating_app_behavior_dataset.csv")

# =====================================================
# DATASET INFO
# =====================================================

total_records = df.shape[0]
total_features = df.shape[1]
model_accuracy = 92.4

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* =====================================================
MAIN APP
===================================================== */

.stApp {
    background: linear-gradient(
        135deg,
        #081120 0%,
        #111827 35%,
        #1d1b4b 70%,
        #ff4f93 100%
    );

    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* =====================================================
REMOVE DEFAULT PADDING
===================================================== */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #1e1b4b 100%
    );

    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* =====================================================
HEADINGS
===================================================== */

.main-title {
    font-size: 68px;
    font-weight: 800;
    color: white;
    margin-bottom: 20px;
    line-height: 1.1;
}

.section-title {
    font-size: 38px;
    font-weight: 700;
    color: white;
    margin-bottom: 20px;
    margin-top: 10px;
}

.sub-text {
    color: #dbe4f3;
    font-size: 20px;
    line-height: 1.9;
}

/* =====================================================
HERO SECTION
===================================================== */

.hero-container {
    background: rgba(255,255,255,0.07);

    backdrop-filter: blur(12px);

    padding: 65px;

    border-radius: 32px;

    border: 1px solid rgba(255,255,255,0.08);

    margin-bottom: 40px;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.30);

    transition: all 0.4s ease;
}

.hero-container:hover {
    transform: translateY(-6px);
    box-shadow: 0px 18px 40px rgba(0,0,0,0.40);
}

/* =====================================================
METRIC CARDS
===================================================== */

.metric-card {
    background: rgba(15,23,42,0.82);

    padding: 35px;

    border-radius: 26px;

    border: 1px solid rgba(255,255,255,0.08);

    text-align: center;

    box-shadow: 0px 8px 25px rgba(0,0,0,0.30);

    transition: all 0.35s ease;

    cursor: pointer;
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);

    background: rgba(30,41,59,0.95);

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow: 0px 15px 35px rgba(255,79,147,0.25);
}

.metric-title {
    font-size: 18px;
    color: #dbe4f3;
    margin-bottom: 15px;
    font-weight: 500;
}

.metric-value {
    font-size: 48px;
    font-weight: 800;
    color: white;
}

/* =====================================================
INFO BOX
===================================================== */

.info-box {
    background: rgba(15,23,42,0.82);

    padding: 42px;

    border-radius: 28px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow: 0px 8px 25px rgba(0,0,0,0.30);

    color: #e5e7eb;

    line-height: 1.9;

    transition: all 0.35s ease;
}

.info-box:hover {
    transform: translateY(-5px);

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow: 0px 15px 35px rgba(59,130,246,0.20);
}

/* =====================================================
WORKFLOW BOX
===================================================== */

.workflow-box {
    background: rgba(30,41,59,0.90);

    padding: 25px;

    border-radius: 20px;

    margin-bottom: 18px;

    border-left: 6px solid #ff4f93;

    font-size: 18px;

    color: white;

    box-shadow: 0px 5px 15px rgba(0,0,0,0.25);

    transition: all 0.35s ease;
}

.workflow-box:hover {
    transform: translateX(8px);

    background: rgba(51,65,85,0.95);

    box-shadow: 0px 10px 25px rgba(255,79,147,0.20);
}

/* =====================================================
DATAFRAME
===================================================== */

[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* =====================================================
STREAMLIT METRICS
===================================================== */

[data-testid="metric-container"] {
    background: rgba(15,23,42,0.82);

    border-radius: 18px;

    padding: 18px;

    border: 1px solid rgba(255,255,255,0.08);

    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-5px);

    border: 1px solid rgba(255,255,255,0.14);

    box-shadow: 0px 10px 20px rgba(255,79,147,0.15);
}

[data-testid="metric-container"] label {
    font-size: 15px !important;
}

[data-testid="metric-container"] div {
    font-size: 18px !important;
}

/* =====================================================
BUTTON HOVER
===================================================== */

.stButton > button {
    background: linear-gradient(
        135deg,
        #ff4f93,
        #8b5cf6
    );

    color: white;

    border: none;

    border-radius: 12px;

    padding: 0.7rem 1.4rem;

    font-weight: 600;

    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.05);

    box-shadow: 0px 10px 25px rgba(255,79,147,0.35);
}

/* =====================================================
TEXT
===================================================== */

p, li {
    font-size: 17px;
    line-height: 1.8;
    color: #e5e7eb;
}

/* =====================================================
SCROLLBAR
===================================================== */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #ff4f93;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("LoveLens")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "Project Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Dashboard Features

- Relationship Prediction
- Dataset Analysis
- Visualization Analytics
- Model Evaluation
- Machine Learning Insights
""")

# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    # HERO SECTION

    st.markdown(
        """
        <style>
        .hero-container{
            background: rgba(255,255,255,0.06);
            padding: 60px;
            border-radius: 30px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 35px;
        }

        .hero-heading{
            font-size: 64px;
            font-weight: 800;
            color: white;
            margin-bottom: 20px;
            line-height: 1.1;
        }

        .hero-text{
            font-size: 22px;
            color: #e2e8f0;
            line-height: 1.8;
            max-width: 1000px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-heading">
                LoveLens Dashboard
            </div>

        <div class="hero-text">
                An intelligent machine learning platform designed to analyze
                relationship compatibility and predict match outcomes using
                behavioral and personality-based features from online dating applications.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Dataset Size</div>
            <div class="metric-value">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Feature Count</div>
            <div class="metric-value">{total_features}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model Accuracy</div>
            <div class="metric-value">{model_accuracy}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # =====================================================
    # PROJECT OBJECTIVE
    # =====================================================

    st.markdown(
        '<div class="section-title">Project Objective</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">

    LoveLens is an interactive machine learning dashboard that predicts
    relationship compatibility using behavioral and personality data.

    <br>

    The system applies machine learning classification techniques to identify
    patterns from user activity and interaction features collected from online
    dating applications.

    <br>

    The dashboard provides prediction capabilities, dataset exploration,
    visualization analytics, and performance evaluation to help users
    understand the behavior behind successful matches.

    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # =====================================================
    # WORKFLOW
    # =====================================================

    st.markdown(
        '<div class="section-title">Machine Learning Workflow</div>',
        unsafe_allow_html=True
    )

    workflow_steps = [
        "Data Collection",
        "Data Preprocessing",
        "Exploratory Data Analysis",
        "Feature Engineering",
        "Model Training",
        "Model Evaluation",
        "Prediction Deployment"
    ]

    for i, step in enumerate(workflow_steps, start=1):

        st.markdown(f"""
        <div class="workflow-box">
            <strong>Step {i}</strong><br><br>
            {step}
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# DATASET OVERVIEW PAGE
# =====================================================

elif page == "Dataset Overview":

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This section provides an overview of the dataset used for relationship compatibility prediction."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Target Variable", "Match Outcome")

    st.write("")

    st.markdown(
        '<div class="section-title">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(df.head(10), use_container_width=True)

    st.write("")

    st.markdown(
        '<div class="section-title">Feature List</div>',
        unsafe_allow_html=True
    )

    feature_df = pd.DataFrame({
        "Feature Name": df.columns
    })

    st.dataframe(feature_df, use_container_width=True)

    st.write("")

    st.markdown(
        '<div class="section-title">Data Types</div>',
        unsafe_allow_html=True
    )

    dtype_df = pd.DataFrame({
        "Column": df.dtypes.index,
        "Data Type": df.dtypes.values
    })

    st.dataframe(dtype_df, use_container_width=True)

    st.write("")

    st.markdown(
        '<div class="section-title">Missing Values</div>',
        unsafe_allow_html=True
    )

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing_df, use_container_width=True)

# =====================================================
# PROJECT INFORMATION PAGE
# =====================================================

elif page == "Project Information":

    st.markdown(
        '<div class="section-title">Project Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">

    <h3>Technologies Used</h3>

    <ul>
        <li>Python</li>
        <li>Streamlit</li>
        <li>Pandas</li>
        <li>Scikit-learn</li>
        <li>Plotly</li>
    </ul>

    <br>

    <h3>Machine Learning Workflow</h3>

    <ol>
        <li>Data Collection</li>
        <li>Data Preprocessing</li>
        <li>Exploratory Data Analysis</li>
        <li>Model Training</li>
        <li>Model Evaluation</li>
        <li>Prediction Dashboard Deployment</li>
    </ol>

    <h3>Team Members</h3>

    <ul>
        <li>Jasmine Chin Ying Hui</li>
        <li>Lee Jian Cheng</li>
        <li>Ng Yue Qhi</li>
        <li>Nurul Afyqah binti Lukman</li>
        <li>Ratu Ahdys Khairany</li>
        <li>Tan Pei Shing</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)