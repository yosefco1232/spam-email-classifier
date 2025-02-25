import streamlit as st
import joblib
from utils import clean_text
import pandas as pd
import os

# Load the model and vectorizer
model = joblib.load('spam_classifier.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Set page title and icon
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered",
)

# Initialize session state variables
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = []
if "history" not in st.session_state:
    st.session_state.history = []
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

# Custom CSS for better readability
st.markdown(
    """
    <style>
    /* Main app background and text color */
    .stApp {
        background-color: #2d2d2d;  /* Dark gray */
        color: #ffffff;  /* White text */
    }

    /* Sidebar background and text color */
    .css-1d391kg {
        background-color: #f0f0f0 !important;  /* Light gray */
    }
    .css-1d391kg p, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #000000 !important;  /* Black text */
    }

    /* Text input and text area styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        color: #ffffff;  /* White text */
        background-color: #1e1e1e;  /* Darker gray */
        border: 1px solid #4CAF50;  /* Green border */
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;  /* Green */
        color: white;  /* White text */
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }

    /* File uploader styling */
    .stFileUploader>div>div>button {
        background-color: #4CAF50;  /* Green */
        color: white;  /* White text */
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stFileUploader>div>div>button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }

    /* Markdown headers and text */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff;  /* White text */
    }
    .stMarkdown p {
        color: #ffffff;  /* White text */
    }

    /* Divider styling */
    .stMarkdown hr {
        border-color: #4CAF50;  /* Green divider */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to save feedback to a file
def save_feedback(email_text, corrected_label):
    feedback_file = "updated_dataset.csv"  # Save directly in the project folder

    # Create a DataFrame with the feedback data
    feedback_data = {"text": [email_text], "label": [corrected_label]}
    feedback_df = pd.DataFrame(feedback_data)

    # Append to the existing file or create a new one
    if os.path.exists(feedback_file):
        feedback_df.to_csv(feedback_file, mode="a", header=False, index=False)
    else:
        feedback_df.to_csv(feedback_file, index=False)

# App title and description
st.title("📧 Spam Email Classifier")
st.markdown("""
    This app uses a machine learning model to classify emails as **spam** or **ham** (not spam).  
    Enter an email in the text box below or upload a text file to get started.
""")

# Sidebar for additional options
with st.sidebar:
    st.header("Settings")
    st.markdown("Customize your experience:")
    st.markdown("---")
    st.markdown("**How to Use:**")
    st.markdown("1. Enter an email in the text box.")
    st.markdown("2. Click **Predict** to classify the email.")
    st.markdown("3. Upload a text file for bulk classification.")

# Input options
input_option = st.radio("Choose input type:", ("Text Input", "File Upload"))

# Text input
if input_option == "Text Input":
    user_input = st.text_area("Enter the email text:", height=200)
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            # Store user input in session state
            st.session_state.user_input = user_input
            st.session_state.show_feedback = True  # Show feedback section

            # Preprocess the input
            cleaned_input = clean_text(user_input)
            input_tfidf = tfidf.transform([cleaned_input])

            # Make prediction
            st.session_state.prediction = model.predict(input_tfidf)[0]
            st.session_state.confidence = model.predict_proba(input_tfidf)[0]

            # Add prediction to history
            st.session_state.history.append({
                "Text": user_input,
                "Prediction": st.session_state.prediction,
                "Confidence": max(st.session_state.confidence),
                "Feedback": ""  # Initialize feedback as empty
            })

# File upload
else:
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        # Read the file
        file_contents = uploaded_file.read().decode("utf-8")
        st.text_area("File Contents:", file_contents, height=200)

        if st.button("Predict"):
            # Store user input in session state
            st.session_state.user_input = file_contents
            st.session_state.show_feedback = True  # Show feedback section

            # Preprocess the input
            cleaned_input = clean_text(file_contents)
            input_tfidf = tfidf.transform([cleaned_input])

            # Make prediction
            st.session_state.prediction = model.predict(input_tfidf)[0]
            st.session_state.confidence = model.predict_proba(input_tfidf)[0]

            # Add prediction to history
            st.session_state.history.append({
                "Text": file_contents,
                "Prediction": st.session_state.prediction,
                "Confidence": max(st.session_state.confidence),
                "Feedback": ""  # Initialize feedback as empty
            })

# Display results and feedback if prediction was made
if st.session_state.show_feedback:
    col1, col2 = st.columns(2)

    # Prediction Result in the first column
    with col1:
        st.subheader("Prediction Result")
        if st.session_state.prediction == "spam":
            st.error(f"This email is classified as: **{st.session_state.prediction.upper()}**")
        else:
            st.success(f"This email is classified as: **{st.session_state.prediction.upper()}**")
        st.write(f"Confidence: {max(st.session_state.confidence):.2f}")

    # Feedback in the second column
    with col2:
        st.subheader("Feedback")
        st.write("Was the prediction correct? If not, please provide the correct label.")
        
        # Two buttons for feedback
        col_spam, col_ham = st.columns(2)
        with col_spam:
            if st.button("Spam"):
                # Update feedback in the history
                st.session_state.history[-1]["Feedback"] = "Spam"
                save_feedback(st.session_state.user_input, "spam")
                st.success("Thank you for your feedback! The dataset has been updated.")
        with col_ham:
            if st.button("Ham"):
                # Update feedback in the history
                st.session_state.history[-1]["Feedback"] = "Ham"
                save_feedback(st.session_state.user_input, "ham")
                st.success("Thank you for your feedback! The dataset has been updated.")

# History section
st.markdown("---")
st.subheader("Prediction History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.style.map(lambda x: "color: red" if x == "spam" else "color: green", subset=["Prediction"]))
else:
    st.info("No predictions made yet.")