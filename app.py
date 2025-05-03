import streamlit as st
import joblib
import pandas as pd
import os

# --- Page config ---
st.set_page_config(page_title="Drug Risk & Side-Effect Predictor", layout="wide")

# --- CSS for layout ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 900px;
            margin: auto;
        }
        h1 {
            margin-bottom: 0.5rem !important;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        .stButton button {
            width: 100%;
            font-size: 16px;
            padding: 0.5rem 1rem;
        }
        .stMarkdown {
            margin-bottom: 0.5rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load models ---
@st.cache_resource
def load_models():
    risk_model = joblib.load(os.path.join("src", "risk_model.pkl"))
    vectorizer_risk = joblib.load(os.path.join("src", "vectorizer_risk.pkl"))
    side_effect_model = joblib.load(os.path.join("src", "side_effect_model.pkl"))
    vectorizer_effects = joblib.load(os.path.join("src", "vectorizer_effects.pkl"))
    mlb = joblib.load(os.path.join("src", "side_effect_mlb.pkl"))
    return risk_model, vectorizer_risk, side_effect_model, vectorizer_effects, mlb

risk_model, vectorizer_risk, side_effect_model, vectorizer_effects, mlb = load_models()

# --- Helper functions ---
def preprocess_input(drug_string):
    return ",".join([token.strip().lower() for token in drug_string.split(",") if token.strip()])

def has_known_drug(drug_string, vectorizer):
    tokens = [token.strip().lower() for token in drug_string.split(",")]
    vocab = set(vectorizer.vocabulary_.keys())
    return any(token in vocab for token in tokens)

def predict_risk_and_effects(raw_input):
    drug_string = preprocess_input(raw_input)
    if not has_known_drug(drug_string, vectorizer_risk):
        return "Unknown", []
    X_input_risk = vectorizer_risk.transform([drug_string])
    risk = risk_model.predict(X_input_risk)[0]
    X_input_effects = vectorizer_effects.transform([drug_string])
    y_pred = side_effect_model.predict(X_input_effects)
    effects = mlb.inverse_transform(y_pred)[0]
    return risk, sorted(effects)

# --- Load CSV for risk label lookup ---
csv_path = os.path.join("src", "Dataset_with_Risk_Label.csv")
try:
    drug_df = pd.read_csv(csv_path)
    drug_df.columns = drug_df.columns.str.strip()
except Exception as e:
    st.error(f"Could not load the CSV file: {e}")
    drug_df = pd.DataFrame()

# --- Get all known drugs for autocomplete ---
all_known_drugs = set()
for combo in drug_df['drug_list']:
    if pd.notnull(combo):
        all_known_drugs.update([drug.strip() for drug in combo.split(',')])
all_known_drugs = sorted(all_known_drugs)

# --- UI: App Title ---
st.markdown("<h1 style='text-align: center;'>🧪 Drug Risk & Side-Effect Predictor</h1>", unsafe_allow_html=True)
st.markdown("Select or type drug names to predict associated risk and side effects.")

# --- Form for prediction with enter support ---
selected_drugs = st.multiselect("Select drugs", options=all_known_drugs)

if selected_drugs:
    user_input = ", ".join(selected_drugs)
    with st.spinner("🔄 Analyzing drugs..."):
        try:
            risk, effects = predict_risk_and_effects(user_input)
            st.markdown(f"### ⚠️ Predicted Risk Level: :green[`{risk}`]" if risk != "Unknown" else f"### ⚠️ Predicted Risk Level: `{risk}`")
            st.markdown("### 💊 Predicted Side Effects:")
            if effects:
                num_cols = 3
                cols = st.columns(num_cols)
                for idx, effect in enumerate(effects):
                    cols[idx % num_cols].markdown(f"- {effect}")
            else:
                st.markdown("_No side effects predicted._")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")


# --- Separator ---
st.markdown("---")

# --- Section 2: Risk label lookup ---
st.markdown("## 🔍 Lookup Drugs by Risk Level")

risk_levels = sorted(drug_df['risk_label'].dropna().unique()) if not drug_df.empty else ['Low', 'Medium', 'High']
selected_risk = st.selectbox("Select a risk level", risk_levels)

if not drug_df.empty:
    rows = drug_df[drug_df['risk_label'].str.lower() == selected_risk.lower()]
    all_drugs = []
    for combo in rows['drug_list']:
        if pd.notnull(combo):
            all_drugs.extend([drug.strip() for drug in combo.split(',')])
    filtered_drugs = sorted(set(all_drugs))
else:
    filtered_drugs = ['aspirin', 'ibuprofen'] if selected_risk.lower() == 'high' else []

if filtered_drugs:
    st.success(f"Drugs with '{selected_risk}' risk label:")
    num_cols = 3
    cols = st.columns(num_cols)
    for idx, drug in enumerate(filtered_drugs):
        cols[idx % num_cols].markdown(f"- {drug}")
else:
    st.warning("No drugs found for the selected risk level.")
