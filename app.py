
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# WineLens: configuration
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

st.set_page_config(
    page_title="WineLens | Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
)

# --------------------------------------------------
# Load the verified models
# --------------------------------------------------

@st.cache_resource
def load_models():
    classifier = joblib.load(MODEL_DIR / "winelens_classifier.joblib")
    regressor = joblib.load(MODEL_DIR / "winelens_regressor.joblib")
    return classifier, regressor


try:
    classifier, regressor = load_models()
except Exception as exc:
    st.error(f"Could not load the models: {exc}")
    st.stop()

# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("🍷 WineLens")
st.subheader("Explore wine quality through machine learning")
st.write(
    "Enter the wine's physicochemical measurements to get predictions "
    "from two Random Forest models trained on the red wine quality dataset."
)

st.info(
    "This is an educational prediction tool. Its estimates are not a "
    "professional wine assessment or a guarantee of perceived quality."
)

# --------------------------------------------------
# Input form
# --------------------------------------------------

DEFAULTS = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}

RANGES = {
    "fixed acidity": (0.0, 20.0),
    "volatile acidity": (0.0, 2.0),
    "citric acid": (0.0, 2.0),
    "residual sugar": (0.0, 100.0),
    "chlorides": (0.0, 1.0),
    "free sulfur dioxide": (0.0, 300.0),
    "total sulfur dioxide": (0.0, 500.0),
    "density": (0.9, 1.1),
    "pH": (0.0, 14.0),
    "sulphates": (0.0, 2.0),
    "alcohol": (0.0, 25.0),
}

with st.form("wine_input_form"):
    st.markdown("### Enter chemical measurements")
    st.caption("Values are prefilled with an example profile. Replace them with your measurements.")

    columns = st.columns(3)
    values = {}

    for index, feature in enumerate(FEATURES):
        minimum, maximum = RANGES[feature]

        with columns[index % 3]:
            values[feature] = st.number_input(
                feature.title(),
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(DEFAULTS[feature]),
                step=0.01,
                format="%.4f" if feature == "density" else "%.3f",
            )

    submitted = st.form_submit_button(
        "Predict wine quality",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Predictions
# --------------------------------------------------

if submitted:
    input_df = pd.DataFrame([values], columns=FEATURES)

    classifier_prediction = int(classifier.predict(input_df)[0])
    regression_prediction = float(regressor.predict(input_df)[0])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Classification prediction")
        st.metric(
            label="Predicted quality category",
            value=classifier_prediction,
        )
        st.caption("The classifier predicts an integer quality category.")

    with col2:
        st.markdown("### Regression prediction")
        st.metric(
            label="Predicted numerical score",
            value=f"{regression_prediction:.2f}",
        )
        st.caption("The regressor estimates a numerical score; decimals are expected.")

    st.markdown("---")
    st.markdown("### Submitted measurements")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.caption(
        "Model performance varies across quality categories. In our test "
        "experiments, rare quality scores were especially difficult to predict."
    )
else:
    st.markdown("---")
    st.markdown("### Ready to explore?")
    st.write(
        "Review the example measurements above, adjust them if needed, "
        "and select **Predict wine quality**."
    )
