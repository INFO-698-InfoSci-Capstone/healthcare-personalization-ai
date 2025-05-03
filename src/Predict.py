import joblib

# Load models and vectorizers
risk_model = joblib.load("risk_model.pkl")
vectorizer_risk = joblib.load("vectorizer_risk.pkl")

side_effect_model = joblib.load("side_effect_model.pkl")
vectorizer_effects = joblib.load("vectorizer_effects.pkl")
mlb = joblib.load("side_effect_mlb.pkl")

# -------------------------------
# Helper: Clean and normalize input
# -------------------------------
def preprocess_input(drug_string):
    return ",".join([token.strip().lower() for token in drug_string.split(",") if token.strip()])

# -------------------------------
# Helper: Check if any drug is in known vocabulary
# -------------------------------
def has_known_drug(drug_string, vectorizer):
    tokens = drug_string.split(",")
    vocab = set(vectorizer.vocabulary_.keys())
    return any(token in vocab for token in tokens)

# -------------------------------
# Main prediction function
# -------------------------------
def predict_risk_and_effects(raw_input):
    drug_string = preprocess_input(raw_input)

    # Check if input contains at least one known drug
    if not has_known_drug(drug_string, vectorizer_risk):
        return "Unknown", []

    # Predict risk
    X_input_risk = vectorizer_risk.transform([drug_string])
    risk = risk_model.predict(X_input_risk)[0]

    # Predict side effects
    X_input_effects = vectorizer_effects.transform([drug_string])
    y_pred = side_effect_model.predict(X_input_effects)
    effects = mlb.inverse_transform(y_pred)[0]

    return risk, sorted(effects)

# -------------------------------
# Terminal Input
# -------------------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter comma-separated drug names (or type 'exit' to quit): ")
        if user_input.lower().strip() == "exit":
            break
        risk, effects = predict_risk_and_effects(user_input)
        print(f"\n🧪 Drugs: {user_input}")
        print(f"⚠️ Predicted Risk Level: {risk}")
        print("💊 Predicted Side Effects:")
        if effects:
            for effect in effects:
                print(f" - {effect}")
        else:
            print("   No side effects predicted.")


