import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Scholarship Recipient Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("scholarship_prediction_model.pkl")

model = load_model()

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🎓 Scholarship Recipient Prediction")
st.markdown(
    "### Enter student information to predict scholarship eligibility"
)

st.divider()

# --------------------------------------------------
# PERSONAL INFORMATION
# --------------------------------------------------
st.header("👤 Personal Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Prefer not to say"]
    )

with col2:
    birth_date = st.text_input(
        "Birth Date",
        placeholder="e.g. 15/08/2002"
    )

with col3:
    birth_city = st.text_input(
        "Birth City",
        placeholder="e.g. Hyderabad"
    )

residence_city = st.text_input(
    "Residence City",
    placeholder="e.g. Hyderabad"
)

# --------------------------------------------------
# UNIVERSITY INFORMATION
# --------------------------------------------------
st.header("🎓 University Information")

col1, col2, col3 = st.columns(3)

with col1:
    application_year = st.number_input(
        "Application Year",
        min_value=2000,
        max_value=2030,
        value=2024
    )

with col2:
    university_type = st.selectbox(
        "University Type",
        ["Public", "Private"]
    )

with col3:
    university_year = st.selectbox(
        "University Year",
        [
            "1", "2", "3", "4",
            "5", "6",
            "Preparatory",
            "Graduate",
            "Master's"
        ]
    )

university_name = st.text_input(
    "University Name",
    placeholder="Enter university name"
)

department = st.text_input(
    "Department",
    placeholder="e.g. Computer Engineering"
)

col1, col2 = st.columns(2)

with col1:
    university_gpa = st.selectbox(
        "University GPA",
        [
            "3.50 - 4.00",
            "3.00 - 3.49",
            "2.50 - 2.99",
            "2.00 - 2.49",
            "1.00 - 1.99",
            "Below 1.00",
            "No GPA"
        ]
    )

with col2:
    evaluation_score = st.number_input(
        "Evaluation Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

# --------------------------------------------------
# HIGH SCHOOL INFORMATION
# --------------------------------------------------
st.header("🏫 High School Information")

col1, col2 = st.columns(2)

with col1:
    high_school_name = st.text_input(
        "High School Name"
    )

with col2:
    high_school_city = st.text_input(
        "High School City"
    )

col1, col2 = st.columns(2)

with col1:
    high_school_type = st.selectbox(
        "High School Type",
        [
            "Science High School",
            "Anatolian High School",
            "Vocational High School",
            "Private High School",
            "Other"
        ]
    )

with col2:
    high_school_grade = st.selectbox(
        "High School Grade",
        [
            "75 - 100",
            "70 - 84",
            "55 - 69",
            "45 - 54",
            "25 - 44",
            "0 - 24",
            "No Grade"
        ]
    )

high_school_department = st.text_input(
    "High School Department"
)

# --------------------------------------------------
# FAMILY INFORMATION
# --------------------------------------------------
st.header("👨‍👩‍👦 Family Information")

col1, col2 = st.columns(2)

with col1:
    mother_education = st.selectbox(
        "Mother's Education",
        [
            "No Education",
            "Primary School",
            "Middle School",
            "High School",
            "University",
            "Master's",
            "Doctorate"
        ]
    )

with col2:
    father_education = st.selectbox(
        "Father's Education",
        [
            "No Education",
            "Primary School",
            "Middle School",
            "High School",
            "University",
            "Master's",
            "Doctorate"
        ]
    )

col1, col2, col3 = st.columns(3)

with col1:
    mother_work = st.selectbox(
        "Mother's Employment",
        ["Yes", "No", "Retired"]
    )

with col2:
    father_work = st.selectbox(
        "Father's Employment",
        ["Yes", "No", "Retired"]
    )

with col3:
    siblings = st.number_input(
        "Number of Siblings",
        min_value=0,
        max_value=20,
        value=2
    )

# --------------------------------------------------
# OTHER INFORMATION
# --------------------------------------------------
st.header("🌟 Activities & Skills")

col1, col2 = st.columns(2)

with col1:
    entrepreneurship = st.selectbox(
        "Entrepreneurship Experience",
        ["Yes", "No"]
    )

with col2:
    english_known = st.selectbox(
        "English Knowledge",
        ["Yes", "No"]
    )

english_level = st.selectbox(
    "English Level",
    ["Beginner", "Intermediate", "Advanced", "None"]
)

club_member = st.selectbox(
    "Member of an Entrepreneurship Club",
    ["Yes", "No"]
)

professional_sport = st.selectbox(
    "Professional Athlete",
    ["Yes", "No"]
)

ngo_member = st.selectbox(
    "Active NGO Member",
    ["Yes", "No"]
)

ngo_project = st.selectbox(
    "Participated in an NGO Project",
    ["Yes", "No"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.divider()

if st.button(
    "🔮 Predict Scholarship",
    use_container_width=True
):

    st.info(
        "The complete 41-feature prediction mapping will be applied here."
    )