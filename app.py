import streamlit as st
import pandas as pd
import joblib


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Scholarship Recipient Prediction",
    page_icon="🎓",
    layout="centered"
)


# ==================================================
# LOAD LIGHTWEIGHT MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("scholarship_prediction_light.pkl")


model = load_model()


# ==================================================
# HEADER
# ==================================================

st.title("🎓 Scholarship Recipient Prediction")

st.write(
    "A Machine Learning application that predicts whether "
    "a student is likely to receive a scholarship."
)

st.info(
    "This application uses a lightweight Random Forest model "
    "trained on the Kaggle Datathon 2024 dataset."
)


# ==================================================
# STUDENT INFORMATION
# ==================================================

st.subheader("Student Information")

col1, col2 = st.columns(2)


with col1:

    university_type = st.selectbox(
        "University Type",
        ["Public", "Private"]
    )

    number_of_siblings = st.number_input(
        "Number of Siblings",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    university_gpa = st.number_input(
        "University GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.0,
        step=0.1
    )


with col2:

    high_school_grade = st.number_input(
        "High School Grade",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


# ==================================================
# CONVERT ENGLISH INPUTS TO ORIGINAL DATA VALUES
# ==================================================

university_type_map = {
    "Public": "Devlet",
    "Private": "Özel"
}

gender_map = {
    "Male": "Erkek",
    "Female": "Kadın"
}


# ==================================================
# PREDICTION
# ==================================================

st.divider()

if st.button(
    "🔮 Predict Scholarship",
    type="primary",
    use_container_width=True
):

    # Create input data using the same feature names
    # used when training the lightweight model.

    input_data = pd.DataFrame({
        "University_Type": [
            university_type_map[university_type]
        ],
        "Number_of_Siblings": [
            number_of_siblings
        ],
        "High_School_Grade": [
            high_school_grade
        ],
        "University_GPA": [
            university_gpa
        ],
        "Gender": [
            gender_map[gender]
        ]
    })


    # --------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    scholarship_probability = probabilities[1]


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "🎉 The model predicts that the student is likely "
            "to receive a scholarship."
        )

    else:

        st.warning(
            "The model predicts that the student is unlikely "
            "to receive a scholarship."
        )


    # --------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------

    st.metric(
        "Scholarship Probability",
        f"{scholarship_probability:.2%}"
    )

    st.progress(
        float(scholarship_probability)
    )


# ==================================================
# MODEL INFORMATION
# ==================================================

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write("**Project:** Scholarship Recipient Prediction Using Machine Learning")

    st.write("**Dataset:** Kaggle Datathon 2024")

    st.write("**Model:** Lightweight Random Forest Classifier")

    st.write("**Deployment Model Size:** Approximately 0.90 MB")

    st.write("**Features Used:** 5")

    st.write("""
    The five features used by the deployment model are:
    
    - University Type
    - Number of Siblings
    - High School Grade
    - University GPA
    - Gender
    """)


# ==================================================
# REAL-WORLD APPLICATION
# ==================================================

with st.expander("🌍 Real-World Application"):

    st.write("""
    This application demonstrates how machine learning can
    support scholarship screening.

    A scholarship organization could use such a system as an
    initial decision-support tool to identify applications that
    may require further review.

    The prediction should not replace official scholarship
    evaluation or human decision-making.
    """)


# ==================================================
# DISCLAIMER
# ==================================================

st.divider()

st.caption(
    "⚠️ This prediction is generated by a machine learning model "
    "and should not be considered an official scholarship decision."
)
