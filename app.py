# Scholarship Recipient Prediction - Streamlit Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Scholarship Recipient Prediction",
    page_icon="🎓",
    layout="centered"
)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("scholarship_prediction_model.pkl")


model = load_model()


# ---------------------------------------------------------
# Page title
# ---------------------------------------------------------

st.title("🎓 Scholarship Recipient Prediction")

st.write(
    "Enter the student's information below to predict "
    "whether the student is likely to receive a scholarship."
)

st.divider()


# ---------------------------------------------------------
# Student Information
# ---------------------------------------------------------

st.subheader("👨‍🎓 Student Information")


col1, col2 = st.columns(2)

with col1:

    university_type = st.selectbox(
        "University Type",
        ["Public", "Private"]
    )

    application_year = st.number_input(
        "Application Year",
        min_value=2020,
        max_value=2030,
        value=2024,
        step=1
    )

    number_of_siblings = st.number_input(
        "Number of Siblings",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )


with col2:

    evaluation_score = st.number_input(
        "Evaluation Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

    university_gpa = st.selectbox(
        "University GPA",
        [
            "3.50 - 4.00",
            "3.00 - 3.50",
            "3.00 - 3.49",
            "2.50 - 3.00",
            "2.50 - 2.99",
            "2.00 - 2.50",
            "1.80 - 2.49",
            "2.50 ve altı",
            "ORTALAMA BULUNMUYOR"
        ]
    )


st.divider()


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button(
    "🔮 Predict Scholarship",
    use_container_width=True
):

    # Create a dictionary containing all model features.
    # Unspecified features are set to NaN so that the
    # trained preprocessing pipeline can impute them.

    student_data = {
        "Basvuru Yili": application_year,
        "Evaluation_Score": evaluation_score,

        "Gender": np.nan,
        "Birth_Date": np.nan,
        "Dogum Yeri": np.nan,
        "Residence_City": np.nan,
        "University_Name": np.nan,
        "University_Type": (
            "Devlet"
            if university_type == "Public"
            else "Özel"
        ),
        "Department": np.nan,
        "University_Year": np.nan,
        "University_GPA": university_gpa,

        "Daha Once Baska Bir Universiteden Mezun Olmus": np.nan,
        "Lise Adi": np.nan,
        "Lise Adi Diger": np.nan,
        "Lise Sehir": np.nan,
        "High_School_Type": np.nan,
        "Lise Bolumu": np.nan,
        "Lise Bolum Diger": np.nan,
        "High_School_Grade": np.nan,

        "Baska Bir Kurumdan Burs Aliyor mu?": np.nan,
        "Burs Aldigi Baska Kurum": np.nan,
        "Baska Kurumdan Aldigi Burs Miktari": np.nan,

        "Anne Egitim Durumu": np.nan,
        "Anne Calisma Durumu": np.nan,
        "Anne Sektor": np.nan,

        "Baba Egitim Durumu": np.nan,
        "Baba Calisma Durumu": np.nan,
        "Baba Sektor": np.nan,

        "Number_of_Siblings": str(number_of_siblings),

        "Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?": np.nan,
        "Uye Oldugunuz Kulubun Ismi": np.nan,

        "Profesyonel Bir Spor Daliyla Mesgul musunuz?": np.nan,
        "Spor Dalindaki Rolunuz Nedir?": np.nan,

        "Aktif olarak bir STK üyesi misiniz?": np.nan,
        "Hangi STK'nin Uyesisiniz?": np.nan,
        "Stk Projesine Katildiniz Mi?": np.nan,

        "Girisimcilikle Ilgili Deneyiminiz Var Mi?": np.nan,
        "Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?": np.nan,

        "Ingilizce Biliyor musunuz?": np.nan,
        "Ingilizce Seviyeniz?": np.nan,

        "Daha Önceden Mezun Olunduysa, Mezun Olunan Üniversite": np.nan
    }

    # Convert to DataFrame
    student_df = pd.DataFrame([student_data])

    # Make sure the columns are in the same order as the
    # model's training features
    expected_columns = model.feature_names_in_

    student_df = student_df.reindex(
        columns=expected_columns
    )

    # Make prediction
    prediction = model.predict(student_df)

    probability = model.predict_proba(student_df)

    scholarship_probability = probability[0][1] * 100


    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.success(
            "🎉 Scholarship Recipient"
        )

    else:

        st.info(
            "Not a Scholarship Recipient"
        )


    st.metric(
        "Scholarship Probability",
        f"{scholarship_probability:.2f}%"
    )


    st.caption(
        "Model: Random Forest | "
        "Accuracy: 95.02% | "
        "F1 Score: 81.71%"
    )
