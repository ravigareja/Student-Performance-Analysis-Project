import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Student Risk System", layout="wide")

# =========================
# CLEAN CSS (MINIMAL UI)
# =========================
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 600;
}
.subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 25px;
}
.student-name {
    font-size: 26px;
    font-weight: 600;
    text-align: center;
    margin: 15px 0;
}
.card {
    padding: 18px;
    border-radius: 10px;
    background-color: ;
    text-align: center;
    border: 1px solid #2e2e2e;
}
.metric {
    font-size: 20px;
    font-weight: 600;
}
.label {
    font-size: 14px;
    color: #aaa;
}
.section {
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
model = joblib.load("student_model.pkl")
data = pd.read_csv("students.csv")

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">Student Performance Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict • Analyze • Improve</div>', unsafe_allow_html=True)

# =========================
# SEARCH
# =========================
display_list = data["student_id"].astype(str) + " - " + data["student_name"]

selected = st.selectbox("Search Student", display_list)

student_id = int(selected.split(" - ")[0])
student = data[data["student_id"] == student_id]

st.markdown(f'<div class="student-name">{student["student_name"].values[0]}</div>', unsafe_allow_html=True)

# =========================
# STUDENT OVERVIEW
# =========================
st.markdown('<div class="section"><b>Student Overview</b></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"""
<div class="card">
<div class="label">Attendance</div>
<div class="metric">{student["attendance"].values[0]}%</div>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="card">
<div class="label">Quiz Average</div>
<div class="metric">{student["quiz_avg"].values[0]}</div>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="card">
<div class="label">Assignment Rate</div>
<div class="metric">{student["assignment_rate"].values[0]}%</div>
</div>
""", unsafe_allow_html=True)

col4.markdown(f"""
<div class="card">
<div class="label">Internal Marks</div>
<div class="metric">{student["internal_marks_1"].values[0]}</div>
</div>
""", unsafe_allow_html=True)

# =========================
# PREPARE DATA
# =========================
X = student.drop(["student_id", "student_name", "final_result"], axis=1)

prediction = model.predict(X)
probability = model.predict_proba(X)

risk = probability[0][1]

# =========================
# PREDICTION RESULT
# =========================
st.markdown('<div class="section"><b>Prediction Result</b></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

col1.metric("Result", "Fail" if prediction[0]==1 else "Pass")
col2.metric("Risk Score", f"{risk:.2f}")

st.progress(risk)

if risk > 0.6:
    st.error("High Risk")
elif risk > 0.3:
    st.warning("Medium Risk")
else:
    st.success("Low Risk")

# =========================
# SHAP (TABLE SAME AS BEFORE)
# =========================
st.markdown('<div class="section"><b>Why this prediction?</b></div>', unsafe_allow_html=True)

explainer = shap.TreeExplainer(model)
shap_output = explainer(X)

values = shap_output.values
class_index = prediction[0]
values = values[:, :, class_index]
values = values[0]

feature_names = X.columns

total = np.sum(np.abs(values))
percentage_values = np.round((values / total) * 100, 2)

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Impact (%)": np.abs(percentage_values)
})

shap_df = shap_df.sort_values(by="Impact (%)", ascending=False)

# ✅ SAME TABLE FORMAT (AS YOU WANTED)
st.dataframe(shap_df, use_container_width=True)

# =========================
# IMPROVED KEY FACTORS
# =========================
st.markdown('<div class="section"><b>Key Factors</b></div>', unsafe_allow_html=True)

top_df = shap_df.head(3)

for i, row in top_df.iterrows():
    feature = row["Feature"].replace("_", " ").title()
    impact = row["Impact (%)"]

    if impact > 25:
        st.error(f"{feature} has a strong negative impact ({impact:.1f}%)")
    elif impact > 15:
        st.warning(f"{feature} has moderate impact ({impact:.1f}%)")
    else:
        st.info(f"{feature} has lower impact ({impact:.1f}%)")