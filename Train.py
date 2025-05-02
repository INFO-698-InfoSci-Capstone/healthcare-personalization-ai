import pandas as pd
import ast
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# 1. Load and preprocess datasets
# -------------------------------
df_train = pd.read_csv("Dataset_with_Risk_Label.csv")
df_toxicity = pd.read_csv("toxicity_with_side_effects_Sample.csv")

# Ensure 'Side_Effects' is always a list
df_toxicity["Side_Effects"] = df_toxicity["Side_Effects"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)

# Combine drug columns into a single 'drug_list' column
drug_cols = ["drug_1", "drug_2", "drug_3", "drug_4"]
df_toxicity["drug_list"] = df_toxicity[drug_cols].fillna("").agg(','.join, axis=1)
df_toxicity["drug_list"] = df_toxicity["drug_list"].str.strip().str.replace(r",+", ",", regex=True).str.strip(",")

# Normalize and clean the 'drug_list' column in both datasets
df_train["drug_list"] = df_train["drug_list"].fillna("").str.lower().str.strip()
df_train = df_train[df_train["drug_list"] != ""]  # Remove rows with empty drug_list

df_toxicity["drug_list"] = df_toxicity["drug_list"].fillna("").str.lower().str.strip()
df_toxicity = df_toxicity[df_toxicity["drug_list"] != ""]  # Remove rows with empty drug_list

# -------------------------------
# 2. Train risk level model
# -------------------------------
vectorizer_risk = CountVectorizer()
X_risk = vectorizer_risk.fit_transform(df_train["drug_list"])
y_risk = df_train["risk_label"]

X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
    X_risk, y_risk, stratify=y_risk, test_size=0.2, random_state=42
)

risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train_risk, y_train_risk)

y_pred_risk = risk_model.predict(X_test_risk)
print("\n📊 Risk Model Evaluation:")
print("Accuracy:", accuracy_score(y_test_risk, y_pred_risk))
print(classification_report(y_test_risk, y_pred_risk))

# -------------------------------
# 3. Train side effect model
# -------------------------------
vectorizer_effects = CountVectorizer()
X_effects = vectorizer_effects.fit_transform(df_toxicity["drug_list"])

mlb = MultiLabelBinarizer()
Y_effects = mlb.fit_transform(df_toxicity["Side_Effects"])

side_effect_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
side_effect_model.fit(X_effects, Y_effects)

y_pred_effects = side_effect_model.predict(X_effects)
print("\n📊 Side Effect Model Evaluation:")
print(classification_report(Y_effects, y_pred_effects, target_names=mlb.classes_))

# -------------------------------
# 4. Save everything
# -------------------------------
joblib.dump(risk_model, "risk_model.pkl")
joblib.dump(vectorizer_risk, "vectorizer_risk.pkl")

joblib.dump(side_effect_model, "side_effect_model.pkl")
joblib.dump(vectorizer_effects, "vectorizer_effects.pkl")
joblib.dump(mlb, "side_effect_mlb.pkl")

# Optional: Save known tokens for validation
joblib.dump(list(vectorizer_risk.vocabulary_.keys()), "known_drugs.pkl")

print("\n✅ Models and vocabulary saved successfully!")
