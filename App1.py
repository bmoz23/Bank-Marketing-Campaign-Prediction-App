import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json
import os

current_dir = os.path.dirname(__file__)
hello_json_path = os.path.join(current_dir, 'hello.json')
machine_json_path = os.path.join(current_dir, 'machine.json')
model_path = os.path.join(current_dir, 'BMO_model.pkl')
data_path = os.path.join(current_dir, 'ADA_DATASET.csv')

# Lottie animation loading function
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)
        
@st.cache(allow_output_mutation=True)
def load_model():
    with st.spinner("Loading model..."):
        return joblib.load(model_path)

@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv(data_path)

@st.cache(allow_output_mutation=True)
def load_animation_h():
    return load_lottie_file(hello_json_path)

@st.cache(allow_output_mutation=True)
def load_animation_m():
    return load_lottie_file(machine_json_path)

# Load animations, model and data
lottie_animation_hello = load_animation_h()
lottie_animation_machine = load_animation_m()
model = load_model()
data = load_data()

feature_info = """
### Feature Information:
- **age**: Age of the client.
- **job**: Type of job.
- **marital**: Marital status.
- **education**: Education level.
- **default**: Has credit in default? (yes, no).
- **housing**: Has housing loan? (yes, no).
- **loan**: Has personal loan? (yes, no).
- **contact**: Contact communication type.
- **month**: Last contact month of year.
- **day_of_week**: Last contact day of the week.
- **duration**: Last contact duration, in seconds.
- **campaign**: number of contacts performed during this campaign and for this client (numeric, includes last contact).
- **pdays**: Number of days that passed by after the client was last contacted from a previous campaign.
- **previous**: Number of contacts performed before this campaign and for this client.
- **poutcome**: Outcome of the previous marketing campaign.
- **emp.var.rate**: Employment variation rate.
- **cons.price.idx**: Consumer price index.
- **cons.conf.idx**: Consumer confidence index.
- **euribor3m**: Euribor 3 month rate.
- **nr.employed**: Number of employees.
- **Feature Creation --> campaign_previous_interaction**: Interaction between campaign and previous contacts.
- **y**: has the client subscribed a term deposit?
"""

# Upload trained model 
# model_filename = 'best_model.pkl'
model = joblib.load(model_path)
data= pd.read_csv(data_path)

# Page selection using sidebar
st.sidebar.title("Page Navigator")
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Visualization", "Machine Learning Prediction Model","Feedback"])

# Feedback Form
if page == "Feedback":
    st.sidebar.title("We value your feedback!")
    name = st.sidebar.text_input("Your Name")
    email = st.sidebar.text_input("Your Email")
    feedback = st.sidebar.text_area("Your Feedback")

    if st.sidebar.button("Submit"):
        if name and email and feedback:
            with open(os.path.join(current_dir, 'feedback.txt'), 'a') as f:
                f.write(f"Name: {name}\nEmail: {email}\nFeedback: {feedback}\n{'-'*40}\n")
            st.sidebar.success("Thank you for your feedback!")
        else:
            st.sidebar.error("Please fill out all fields.")

if page == "Home":
   st.title("Welcome to the Bank Marketing Campaign Prediction App")
   col1, col2 = st.columns(2)
   with col1:
    st.write("""
            This application allows you to predict the success of a bank marketing campaign 
            based on various features. Use the sidebar to navigate to the prediction page 
            and input the required features to get a prediction.
        """)
    
    #user_name = st.text_input("Can you enter your name here so I can address you?")
    #if user_name:
    #    st.write(f"Hello, {user_name}! Welcome to the app.")
    #    st.write("You can go to the navigation bar to explore the project we have created. Let's continue!")
    with col2:
        st_lottie(lottie_animation_hello, height=300, key="coding")

elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("""
        Welcome to the 'Data Visualization' page. This page is dedicated to the Bank Marketing Data Set used in training our model. Here, you can explore the dataset and examine all the features and data within it. Furthermore, you can also explore various graphical representations of these features.
    """)
    num_rows = st.number_input("Select number of rows to view", min_value=5, max_value=50, value=10)
     # Show the first few rows of the data set
    st.write(f"Here is a preview of the first {num_rows} rows of the dataset:")
    st.write(data.head(num_rows))
    st.write(feature_info)

    # Data Visualization part for our dataset
    st.write("### Distribution of Age")
    fig, ax = plt.subplots()
    sns.histplot(data['age'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.write("### Job Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data['job'], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("### Marital Status Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data['marital'], ax=ax)
    st.pyplot(fig)

    st.write("### Education Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data['education'], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))  
    numerical_data = data.select_dtypes(include=['float64', 'int64']) 
    corr = numerical_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("### Duration vs Age")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='age', y='duration', ax=ax)
    st.pyplot(fig)

    st.write("### Box Plot of Campaign Outcome by Job Type")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='job', y='duration', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "Machine Learning Prediction Model":

    st.title("Bank Marketing Campaign Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""For model selection, we first completed steps such as data preprocessing and data cleaning. We then worked with five different models: Logistic Regression, KNN, Decision Tree, Random Forest, and Gradient Boosting. After evaluating the models based on their F1 scores and accuracy scores, we decided to proceed with the Decision Tree model. We incorporated this model into a pipeline that includes all the steps taken and prepared it for use.""")

    with col2:
        st_lottie(lottie_animation_machine, height=300, key="coding")
        
    with st.expander("Try the model"):
        st.write("Enter the feature values to make a prediction:")

        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
        marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
        education = st.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"])
        default = st.selectbox("Default", ["yes", "no", "unknown"])
        housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"])
        loan = st.selectbox("Personal Loan", ["yes", "no", "unknown"])
        contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
        month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        day_of_week = st.selectbox("Last Contact Day of the Week", ["mon", "tue", "wed", "thu", "fri"])
        duration = st.number_input("Duration of Last Contact (seconds)", min_value=0, value=0)
        pdays = st.number_input("Days Since Last Contact", min_value=-1, value=0)
        poutcome = st.selectbox("Outcome of Previous Marketing Campaign", ["failure", "nonexistent", "success"])
        emp_var_rate = st.number_input("Employment Variation Rate", value=0.0)
        cons_price_idx = st.number_input("Consumer Price Index", value=0.0)
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=0.0)
        euribor3m = st.number_input("Euribor 3 Month Rate", value=0.0)
        nr_employed = st.number_input("Number of Employees", value=0.0)
        campaign_previous_interaction = st.number_input("Campaign Previous Interaction", min_value=0, value=0)

        # Features of data set used during training 
        input_data = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'duration': duration,
            'pdays': pdays,
            'poutcome': poutcome,
            'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m,
            'nr.employed': nr_employed,
            'campaign_previous_interaction': campaign_previous_interaction
        }
        input_df = pd.DataFrame([input_data])

        # Prediction button
        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")

if __name__ == "__main__":
    st.write("Bank Marketing Campaign Prediction")
