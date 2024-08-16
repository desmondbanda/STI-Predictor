import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
data = pd.read_csv('sti_data.csv')

# Preprocessing
le = LabelEncoder()
data['sti_name'] = le.fit_transform(data['sti_name'])

# Define the features and target
X = data.drop(columns=['sti_name', 'sti_symptoms'])
y = data['sti_name']

# Define the preprocessing for numeric and categorical features
numeric_features = ['age', 'body_temp', 'white_blood_cell_count']
categorical_features = ['sexually_active', 'partner_history', 'infection_location', 'antibody_test', 'antibiotic_treatment']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Function to get recommendations, remedies, and video for each STI
def get_sti_info(sti_name):
    recommendations = {
        'Chlamydia': 'Practice safe sex, get tested regularly, antibiotics are usually prescribed for treatment.',
        'Gonorrhea': 'Antibiotics are the primary treatment, avoid unprotected sex.',
        'Syphilis': 'Penicillin is the preferred treatment, avoid sexual contact until fully treated.',
        'HPV': 'Vaccination is available, regular screening is important.',
        'HIV': 'Antiretroviral therapy (ART) is the standard treatment, practice safe sex and use protection.'
        # Add more as needed
    }
    
    remedies = {
        'Chlamydia': 'Drink plenty of water, avoid sexual contact until treatment is complete.',
        'Gonorrhea': 'Avoid alcohol and caffeine, use warm compresses on affected areas.',
        'Syphilis': 'Apply antibiotic ointment to sores, maintain good hygiene.',
        'HPV': 'Boost immune system with vitamin C, zinc, and echinacea supplements.',
        'HIV': 'Eat a balanced diet, exercise regularly, manage stress levels.'
        # Add more as needed
    }
    
    video_urls = {
        'Chlamydia': 'https://www.youtube.com/watch?v=gVH1gY05MsA',
        'Gonorrhea': 'https://www.youtube.com/watch?v=gVH1gY05MsA',
        'Syphilis': 'https://www.youtube.com/watch?v=gVH1gY05MsA',
        'HPV': 'https://www.youtube.com/watch?v=gVH1gY05MsA',
        'HIV': 'https://www.youtube.com/watch?v=gVH1gY05MsA'
        # Add more as needed
    }
    
    return recommendations.get(sti_name, ''), remedies.get(sti_name, ''), video_urls.get(sti_name, '')

# Streamlit app
st.set_page_config(
    page_title='STI Prediction App',
    page_icon=':microscope:',
    layout='wide'
)

# Custom CSS styles
custom_styles = """
<style>
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
    color: #333333;
}
h1 {
    color: #1f77b4;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(custom_styles, unsafe_allow_html=True)

# Sidebar with input fields
st.sidebar.header('Fill in the symptoms')

def user_input_features():
    age = st.sidebar.slider('Age', int(X['age'].min()), int(X['age'].max()), int(X['age'].mean()), step=1)
    body_temp = st.sidebar.slider('Body Temperature', float(X['body_temp'].min()), float(X['body_temp'].max()), float(X['body_temp'].mean()))
    white_blood_cell_count = st.sidebar.slider('White Blood Cell Count', int(X['white_blood_cell_count'].min()), int(X['white_blood_cell_count'].max()), int(X['white_blood_cell_count'].mean()), step=1)
    sexually_active = st.sidebar.selectbox('Sexually Active', ('Yes', 'No'))
    partner_history = st.sidebar.selectbox('Partner History', ('Yes', 'No'))
    infection_location = st.sidebar.selectbox('Infection Location', ('Genital', 'Oral', 'Anal', 'Other'))
    antibody_test = st.sidebar.selectbox('Antibody Test', ('Positive', 'Negative'))
    antibiotic_treatment = st.sidebar.selectbox('Antibiotic Treatment', ('Yes', 'No'))
    data = {
        'age': age,
        'body_temp': body_temp,
        'white_blood_cell_count': white_blood_cell_count,
        'sexually_active': sexually_active,
        'partner_history': partner_history,
        'infection_location': infection_location,
        'antibody_test': antibody_test,
        'antibiotic_treatment': antibiotic_treatment
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Predictions
if st.sidebar.button('Predict'):
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    predicted_sti = le.inverse_transform(prediction)[0]
    st.write(predicted_sti)
    
    # Get additional information
    recommendations, remedies, video_url = get_sti_info(predicted_sti)
    
    # Display recommendations and remedies
    st.subheader('Recommendations')
    st.write(recommendations)
    
    st.subheader('Home Remedies')
    st.write(remedies)
    
    # Display YouTube video
    st.subheader('Information Video')
    if video_url:
        st.video(video_url)
    else:
        st.write('Video not available')
