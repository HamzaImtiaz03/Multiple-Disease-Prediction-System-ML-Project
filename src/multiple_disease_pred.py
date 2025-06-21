import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models


diabetes_model = pickle.load(open('E:/Multiple_Disease_Prediction/saved_models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('E:/Multiple_Disease_Prediction/saved_models/heart_model.sav', 'rb'))
thyroid_model = pickle.load(open('E:/Multiple_Disease_Prediction/saved_models/thyroid_model.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Thyroid Disease Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)
    


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')

    with col2:
        Glucose = st.number_input('Glucose Level')

    with col3:
        BloodPressure = st.number_input('Blood Pressure value')

    with col1:
        SkinThickness = st.number_input('Skin Thickness value')

    with col2:
        Insulin = st.number_input('Insulin Level')

    with col3:
        BMI = st.number_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.number_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Thyroid Disease Prediction Page

def preprocess_input(input_list):
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # Manually encoded based on training-time label encoding
    mappings = {
        'Gender': {'F': 0, 'M': 1},
        'Smoking': {'No': 0, 'Yes': 1},
        'Hx Smoking': {'No': 0, 'Yes': 1},
        'Hx Radiotherapy': {'No': 0, 'Yes': 1},
        'Thyroid Function': {
            'Euthyroid': 0,
            'Clinical Hyperthyroidism': 1,
            'Clinical Hypothyroidism': 2
        },
        'Physical Examination': {
            'Single nodular goiter-left': 0,
            'Single nodular goiter-right': 1,
            'Multinodular goiter': 2
        },
        'Adenopathy': {'No': 0, 'Yes': 1},
        'Pathology': {'Micropapillary': 0},
        'Focality': {'Uni-Focal': 0, 'Multi-Focal': 1},
        'Risk': {'Low': 0},  # Extend if needed
        'T': {'T1a': 0},     # Extend if needed
        'N': {'N0': 0},      # Extend if needed
        'M': {'M0': 0},      # Extend if needed
        'Stage': {'I': 0},   # Extend if needed
        'Response': {'Excellent': 0, 'Indeterminate': 1},
        'Recurred': {'No': 0, 'Yes': 1}
    }

    # Convert values according to above mapping
    processed = []
    feature_names = [
        'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiotherapy',
        'Thyroid Function', 'Physical Examination', 'Adenopathy',
        'Pathology', 'Focality', 'Risk', 'T', 'N', 'M',
        'Stage', 'Response', 'Recurred'
    ]

    for val, feature in zip(input_list, feature_names):
        if feature == 'Age':
            processed.append(float(val))
        else:
            mapped = mappings.get(feature, {}).get(val)
            if mapped is None:
                raise ValueError(f"Invalid value '{val}' for feature '{feature}'")
            processed.append(mapped)

    return np.array(processed)



if selected == 'Thyroid Disease Prediction':
    st.title('Thyroid Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        gender = st.selectbox('Gender', ['F', 'M'])

    with col3:
        smoking = st.selectbox('Smoking', ['Yes', 'No'])

    with col1:
        hx_smoking = st.selectbox('Hx Smoking', ['Yes', 'No'])

    with col2:
        hx_radiotherapy = st.selectbox('Hx Radiotherapy', ['Yes', 'No'])

    with col3:
        thyroid_function = st.selectbox('Thyroid Function', [
            'Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism'
        ])

    with col1:
        physical_exam = st.selectbox('Physical Examination', [
            'Single nodular goiter-left', 'Single nodular goiter-right', 'Multinodular goiter'
        ])

    with col2:
        adenopathy = st.selectbox('Adenopathy', ['Yes', 'No'])

    with col3:
        pathology = st.selectbox('Pathology', ['Micropapillary'])

    with col1:
        focality = st.selectbox('Focality', ['Uni-Focal', 'Multi-Focal'])

    with col2:
        risk = st.selectbox('Risk', ['Low'])

    with col3:
        t_stage = st.selectbox('T', ['T1a'])  # Add other values as needed

    with col1:
        n_stage = st.selectbox('N', ['N0'])  # Add other values as needed

    with col2:
        m_stage = st.selectbox('M', ['M0'])  # Add other values as needed

    with col3:
        stage = st.selectbox('Stage', ['I'])  # Add other values as needed

    with col1:
        response = st.selectbox('Response', ['Excellent', 'Indeterminate'])

    with col2:
        recurred = st.selectbox('Recurred', ['Yes', 'No'])

    thyroid_diagnosis = ''

    if st.button('Thyroid Disease Test Result'):
        user_input = [
            age, gender, smoking, hx_smoking, hx_radiotherapy, thyroid_function, physical_exam,
            adenopathy, pathology, focality, risk, t_stage, n_stage, m_stage,
            stage, response, recurred
        ]

        # TODO: Apply the same encoding you used during training
        # Example: Label encoding, one-hot encoding, or any other preprocessing

        try:
            # Example placeholder for processed features:
            processed_input = preprocess_input(user_input)  # You must define this function

            thyroid_prediction = thyroid_model.predict([processed_input])

            if thyroid_prediction[0] == 1:
                thyroid_diagnosis = 'The person is likely to have thyroid cancer'
            else:
                thyroid_diagnosis = 'The person is unlikely to have thyroid cancer'

        except Exception as e:
            thyroid_diagnosis = f'Error during prediction: {e}'

    st.success(thyroid_diagnosis)


   