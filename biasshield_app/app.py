import streamlit as st
import shap
import pickle
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load Model & Vectorizer ---
with open("models/toxicity_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- SHAP Explainer (cached for session)
@st.cache_resource
def get_shap_explainer():
    dummy_input = vectorizer.transform(["placeholder"])
    return shap.Explainer(model, dummy_input)

explainer = get_shap_explainer()

# --- PDF Generator
def generate_pdf(prompt, score, shap_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "BiasShield Toxicity Report", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Prompt:\n{prompt}")
    pdf.cell(0, 10, f"Toxicity Score: {round(score*100, 2)}%", ln=True)
    if os.path.exists(shap_path):
        pdf.image(shap_path, w=180)
    output_path = f"pdf_reports/report_{uuid.uuid4().hex}.pdf"
    pdf.output(output_path)
    return output_path

# --- Streamlit UI ---
st.set_page_config(page_title="BiasShield", page_icon="🛡️")
st.title("🛡️ BiasShield – AI Toxicity & Bias Analyzer")

st.markdown("Enter a sentence or prompt below. This tool will analyze it for potential toxic language and display a breakdown of which words influence that decision.")

prompt = st.text_area("Enter a prompt", height=120)

if st.button("🔍 Analyze"):
    if not prompt.strip():
        st.warning("Please enter a prompt to analyze.")
    else:
        # Predict
        vec = vectorizer.transform([prompt])
        score = model.predict_proba(vec)[0][1]

        # SHAP explanation
        shap_values = explainer(vec)
        shap_path = f"shap_explanations/force_{uuid.uuid4().hex}.png"
        shap.plots.text(shap_values[0], display=False, matplotlib=True)
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()

        # Display results
        st.markdown("### 🧠 Toxicity Prediction")
        st.metric("Toxicity Score", f"{round(score * 100, 2)}%")

        if score > 0.7:
            st.error("🔴 High Risk – Likely toxic language detected.")
        elif score > 0.4:
            st.warning("🟠 Moderate Risk – May contain questionable language.")
        else:
            st.success("🟢 Low Risk – Clean and non-toxic language.")

        st.markdown("### 🔬 Explanation")
        st.image(shap_path, caption="SHAP Explanation", use_column_width=True)

        # Generate PDF
        report_path = generate_pdf(prompt, score, shap_path)
        with open(report_path, "rb") as f:
            st.download_button("📥 Download PDF Report", data=f, file_name="toxicity_report.pdf")

        st.markdown("---")
        st.markdown("BiasShield uses SHAP + Logistic Regression to interpret toxic content in real time. Ideal for LLM auditing, moderation, and research.")

