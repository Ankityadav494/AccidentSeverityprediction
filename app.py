#!/usr/bin/env python3
"""
Streamlit app for Road Accident Severity Prediction (simple 10-feature version)
Reuses existing SimpleAccidentPredictor from src/simple_prediction.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure we can import from src/
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simple_prediction import SimpleAccidentPredictor, create_sample_scenarios  # noqa: E402


@st.cache_resource(show_spinner=True)
def load_predictor() -> SimpleAccidentPredictor:
    return SimpleAccidentPredictor(models_dir=str(BASE_DIR / "models"))


def get_categorical_options(predictor: SimpleAccidentPredictor) -> dict:
    """Build options for categorical inputs from saved label encoders when available."""
    default_options = {
        "Country": [
            "USA",
            "UK",
            "Canada",
            "Australia",
            "India",
            "Germany",
        ],
        "Month": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        "Day of Week": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        "Time of Day": ["Morning", "Afternoon", "Evening", "Night"],
        "Urban/Rural": ["Urban", "Rural"],
        "Road Type": ["Street", "Highway", "Avenue", "Rural Road"],
        "Weather Conditions": [
            "Clear",
            "Cloudy",
            "Rain",
            "Fog",
            "Snow-covered",
            "Storm",
        ],
    }

    options = {}
    for col, fallback in default_options.items():
        try:
            if col in predictor.label_encoders and hasattr(
                predictor.label_encoders[col], "classes_"
            ):
                # Use learned classes as options; convert to list of strings
                learned = [str(c) for c in predictor.label_encoders[col].classes_]
                # Some datasets store ints; ensure non-empty
                options[col] = learned if len(learned) > 0 else fallback
            else:
                options[col] = fallback
        except Exception:
            options[col] = fallback
    return options


def main() -> None:
    st.set_page_config(
        page_title="Road Accident Severity - Simple Model",
        page_icon="🚗",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # --- Custom CSS ---
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .app-title { font-size: 2.1rem; font-weight: 700; margin-bottom: .25rem; }
        .app-sub { color: #666; margin-bottom: 1rem; }
        .badge { display:inline-block; padding: .35rem .6rem; border-radius: 999px; font-weight:600; }
        .badge-minor { background:#e8f5e9; color:#1b5e20; }
        .badge-moderate { background:#fff3e0; color:#e65100; }
        .badge-major { background:#ffebee; color:#b71c1c; }
        .badge-fatal { background:#ffebee; color:#880e4f; }
        .stButton>button { border-radius:10px; padding:.6rem 1rem; font-weight:600; }
        .section { padding: .75rem 1rem; border: 1px solid #eee; border-radius: 10px; background: #fafafa; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="app-title">🚗 Road Accident Severity Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Simple model using 10 essential features</div>', unsafe_allow_html=True)

    predictor = load_predictor()
    options = get_categorical_options(predictor)

    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.write(
            "This app predicts accident severity using a trained Random Forest/Decision Tree on 10 key features."
        )
        st.write("Artifacts are loaded from the local 'models/' directory.")
        st.markdown("Made with ❤️ using Streamlit")

    # Quick actions row
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("🔄 Reset"):
        for k in [
            "country","year","month","day_of_week","time_of_day","urban_rural",
            "road_type","weather","visibility","vehicles","model_choice"
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    samples = create_sample_scenarios()
    if a2.button("✨ Low-risk sample"):
        s = samples[0]["data"]
        st.session_state.update({
            "country": s["Country"],
            "year": int(s["Year"]),
            "month": s["Month"],
            "day_of_week": s["Day of Week"],
            "time_of_day": s["Time of Day"],
            "urban_rural": s["Urban/Rural"],
            "road_type": s["Road Type"],
            "weather": s["Weather Conditions"],
            "visibility": int(s["Visibility Level"]),
            "vehicles": int(s["Number of Vehicles Involved"]),
        })
        st.rerun()
    if a3.button("⚠️ High-risk sample"):
        s = samples[1]["data"]
        st.session_state.update({
            "country": s["Country"],
            "year": int(s["Year"]),
            "month": s["Month"],
            "day_of_week": s["Day of Week"],
            "time_of_day": s["Time of Day"],
            "urban_rural": s["Urban/Rural"],
            "road_type": s["Road Type"],
            "weather": s["Weather Conditions"],
            "visibility": int(s["Visibility Level"]),
            "vehicles": int(s["Number of Vehicles Involved"]),
        })
        st.rerun()

    # Model selector
    available_models = predictor.get_available_models() or ["random_forest"]
    default_model_idx = 0 if "random_forest" in available_models else 0
    model_choice = st.selectbox(
        "Model",
        available_models,
        index=default_model_idx,
        key="model_choice",
        help="Choose which trained model to use",
    )

    with st.form("prediction_form"):
        st.markdown("### 🧾 Inputs")

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            country = st.selectbox("🌎 Country", options["Country"], index=0, key="country")
        with c2:
            year = st.number_input("📅 Year", min_value=1990, max_value=2100, value=st.session_state.get("year", 2023), step=1, key="year")
        with c3:
            month = st.selectbox("🗓️ Month", options["Month"], index=5, key="month")

        c4, c5, c6 = st.columns(3)
        with c4:
            day_of_week = st.selectbox("📆 Day of Week", options["Day of Week"], index=0, key="day_of_week")
        with c5:
            time_of_day = st.selectbox("🕒 Time of Day", options["Time of Day"], index=0, key="time_of_day")
        with c6:
            urban_rural = st.selectbox("🏙️/🌾 Urban/Rural", options["Urban/Rural"], index=0, key="urban_rural")

        c7, c8, c9 = st.columns(3)
        with c7:
            road_type = st.selectbox("🛣️ Road Type", options["Road Type"], index=0, key="road_type")
        with c8:
            weather = st.selectbox("🌦️ Weather Conditions", options["Weather Conditions"], index=0, key="weather")
        with c9:
            visibility = st.number_input(
                "👁️ Visibility (m)", min_value=0, max_value=10000, value=st.session_state.get("visibility", 200), step=10, key="visibility"
            )

        vehicles = st.number_input(
            "🚘 Vehicles Involved", min_value=1, max_value=20, value=st.session_state.get("vehicles", 1), step=1, key="vehicles"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict Severity")

    if submitted:
        scenario = {
            "Country": country,
            "Year": int(year),
            "Month": month,
            "Day of Week": day_of_week,
            "Time of Day": time_of_day,
            "Urban/Rural": urban_rural,
            "Road Type": road_type,
            "Weather Conditions": weather,
            "Visibility Level": int(visibility),
            "Number of Vehicles Involved": int(vehicles),
        }

        with st.spinner("Predicting..."):
            result = predictor.predict_severity(scenario, model_name="random_forest")

        if result is None:
            st.error("Prediction failed. Please try different inputs or reload the app.")
            return

        severity = result["predicted_severity"]
        scores = result["confidence_scores"]

        # Severity badge
        badge_class = {
            "Minor": "badge badge-minor",
            "Moderate": "badge badge-moderate",
            "Major": "badge badge-major",
            "Fatal": "badge fatal badge-fatal",
        }.get(severity, "badge")
        st.markdown(
            f"<span class='{badge_class}'>Predicted: {severity}</span>",
            unsafe_allow_html=True,
        )

        # Confidence bars
        prob_df = pd.DataFrame(
            [{"Severity": k, "Confidence": float(v)} for k, v in scores.items()]
        ).sort_values("Confidence", ascending=False)
        st.bar_chart(prob_df.set_index("Severity"), height=220)

        with st.expander("Details"):
            st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
            st.write("Model used:", model_choice)

        # Helpful interpretation
        interpretations = {
            "Minor": "Low-risk situation with minimal damage expected.",
            "Moderate": "Moderate risk; some factors increase caution.",
            "Major": "High risk; multiple risk factors present.",
            "Fatal": "Very high risk; severe consequences possible.",
        }
        st.info(interpretations.get(severity, "Interpretation unavailable."))

        # Feature importance (if available)
        fi = predictor.get_feature_importance(model_name=model_choice)
        if fi is not None and not fi.empty:
            st.markdown("### 🔍 Feature Importance")
            st.bar_chart(fi.set_index("feature"), height=300)


if __name__ == "__main__":
    main()


