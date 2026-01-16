import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Try importing plotly, handle if missing
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Page Config
st.set_page_config(
    page_title="OncoMinds | Advanced Breast Cancer Analytics",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) - Premium Dark Theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1f2937;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #ff4b4b !important;
    }
    
    /* Cards */
    .metric-card {
        background-color: #262730;
        border: 1px solid #41424b;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data
def load_data():
    """Load and cache the breast cancer dataset."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    return df, data.feature_names

@st.cache_resource
def train_model(df, feature_names):
    """Train and cache the Random Forest model."""
    X = df[feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, scaler, acc

# --- App Initialization ---

df, feature_names = load_data()
model, scaler, model_accuracy = train_model(df, feature_names)

# --- Sidebar Navigation ---

st.sidebar.markdown("# 🎗️ OncoMinds")
st.sidebar.markdown("### Real-time Diagnostic System")
st.sidebar.markdown("---")

selection = st.sidebar.radio(
    "Navigate", 
    ["Dashboard", "Data Analysis", "Predictive Analytics", "Upload Data"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"🦾 Model Accuracy: **{model_accuracy:.2%}**")
st.sidebar.markdown("Using RandomForest Classifier")

# --- Page Content ---

if selection == "Dashboard":
    st.title("🛡️ Executive Dashboard")
    st.markdown("### Breast Cancer Dataset Overview")
    st.markdown("Real-time view of the dataset statistics and class distribution.")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{df.shape[0]:,}")
    with col2:
        st.metric("Features Tracked", f"{df.shape[1]-2}")
    with col3:
        malignant_count = df[df['target'] == 0].shape[0]
        st.metric("Malignant Cases", f"{malignant_count}", delta="High Risk", delta_color="inverse")
    with col4:
        benign_count = df[df['target'] == 1].shape[0]
        st.metric("Benign Cases", f"{benign_count}")

    st.markdown("---")
    
    # Charts
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Diagnosis Distribution")
        if HAS_PLOTLY:
            fig = px.pie(df, names='diagnosis', 
                         color='diagnosis', 
                         color_discrete_map={'Malignant':'#ef4444', 'Benign':'#3b82f6'},
                         hole=0.5)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                              font=dict(color="white", size=14),
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df['diagnosis'].value_counts())

    with c2:
        st.subheader("Quick Stats")
        st.dataframe(df.describe().T[['mean', 'std', 'min', 'max']], height=400)

elif selection == "Data Analysis":
    st.title("📊 Advanced Data Analytics")
    
    tab1, tab2 = st.tabs(["Correlation Engine", "Feature Inspector"])
    
    with tab1:
        st.subheader("Feature Correlation Heatmap")
        st.markdown("Analyze relationships between different clinical features.")
        fig, ax = plt.subplots(figsize=(20, 15))
        sns.heatmap(df[feature_names].corr(), annot=False, cmap="magma", linewidths=0.5, linecolor='gray', ax=ax)
        # Dark mode adjustment for matplotlib
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Feature Distribution Analysis")
        selected_feature = st.selectbox("Select Feature to Analyze", feature_names)
        
        if HAS_PLOTLY:
            fig = px.histogram(df, x=selected_feature, color="diagnosis", marginal="box",
                               color_discrete_map={'Malignant':'#ef4444', 'Benign':'#3b82f6'},
                               nbins=50,
                               title=f"Distribution of {selected_feature} by Diagnosis")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                              font=dict(color="white"),
                              xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df[selected_feature])

elif selection == "Predictive Analytics":
    st.title("🤖 AI Diagnostic Tool")
    st.markdown("### Real-time Risk Assessment Model")
    st.markdown("Enter patient clinical metrics below to generate a diagnosis prediction.")
    
    with st.form("prediction_form"):
        st.subheader("Clinical Metrics Input")
        
        # Use columns for a grid layout inputs
        cols = st.columns(3)
        input_data = {}
        
        # Split features into 3 columns
        chunks = np.array_split(feature_names, 3)
        
        for i, col in enumerate(cols):
            with col:
                for feature in chunks[i]:
                    # nice formatting for label, maybe remove units for cleaner look or keep them
                    label = feature.replace('_', ' ').title()
                    val = st.number_input(label, 
                                          value=float(df[feature].mean()), 
                                          step=0.1,
                                          format="%.4f")
                    input_data[feature] = val
        
        st.markdown("---")
        submit = st.form_submit_button("Run Diagnostic Analysis", use_container_width=True)
    
    if submit:
        # Prepare input
        input_df = pd.DataFrame([input_data])
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.markdown("### Analysis Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 0:
                st.error("## ⚠️ Malignant Detected")
                st.markdown(f"**Confidence Level:** {probability[0]:.2%}")
                st.markdown("Recommendation: Immediate clinical consultation advised.")
            else:
                st.success("## ✅ Benign Indication")
                st.markdown(f"**Confidence Level:** {probability[1]:.2%}")
                st.markdown("Recommendation: Routine follow-up.")
        
        with res_col2:
            st.markdown("#### Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': ['Malignant', 'Benign'],
                'Probability': probability
            })
            
            if HAS_PLOTLY:
                fig = px.bar(prob_df, x='Probability', y='Class', orientation='h', 
                             color='Class', 
                             color_discrete_map={'Malignant':'#ef4444', 'Benign':'#3b82f6'},
                             text_auto='.2%')
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                                  font=dict(color="white"),
                                  xaxis=dict(range=[0, 1], showgrid=False),
                                  yaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.table(prob_df)

elif selection == "Upload Data":
    st.title("📂 Upload External Data")
    st.markdown("Run the analysis pipeline on your own batch of data from a `.csv` file.")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        uploaded_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        st.subheader("Data Overview")
        st.dataframe(uploaded_data.head())
        
        if st.checkbox("Show Correlation Matrix"):
             fig, ax = plt.subplots(figsize=(10, 8))
             sns.heatmap(uploaded_data.corr(), cmap="coolwarm", ax=ax)
             st.pyplot(fig)

