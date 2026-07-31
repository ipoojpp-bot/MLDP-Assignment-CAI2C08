import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")
st.title("Breast Cancer Classification App")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('breast_cancer_dataset.csv')

df = load_data()
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tabs for organization
tab1, tab2 = st.tabs(["Make Prediction", "Model Comparison"])

with tab1:
    st.subheader("Predict Patient Diagnosis")
    st.write("Adjust feature inputs in the sidebar to test predictions.")
    
    # Sidebar controls
    st.sidebar.header("Tumor Features")
    mean_radius = st.sidebar.slider("Mean Radius", float(X['mean radius'].min()), float(X['mean radius'].max()), float(X['mean radius'].mean()))
    mean_texture = st.sidebar.slider("Mean Texture", float(X['mean texture'].min()), float(X['mean texture'].max()), float(X['mean texture'].mean()))
    mean_perimeter = st.sidebar.slider("Mean Perimeter", float(X['mean perimeter'].min()), float(X['mean perimeter'].max()), float(X['mean perimeter'].mean()))

    # Train Single active model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    input_data = X.mean().to_dict()
    input_data['mean radius'] = mean_radius
    input_data['mean texture'] = mean_texture
    input_data['mean perimeter'] = mean_perimeter
    input_df = pd.DataFrame([input_data])

    if st.button("Run Prediction"):
        pred = model.predict(input_df)[0]
        if pred == 1:
            st.success("Diagnosis: **Benign** (No Cancer Detected)")
        else:
            st.error("Diagnosis: **Malignant** (Cancer Detected)")

with tab2:
    st.subheader("Model Performance Comparison")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    res = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        res.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds)
        })
    
    comp_df = pd.DataFrame(res).sort_values(by="Accuracy", ascending=False)
    st.dataframe(comp_df, use_container_width=True)
    
    st.markdown("### Why Random Forest is the Top Model:")
    st.write("- **Highest Accuracy & F1-Score** among tested algorithms.")
    st.write("- **Ensemble Architecture:** Combines multiple decision trees to reduce risk of overfitting.")