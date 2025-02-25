# Spam Email Classifier

## Overview
This project is a machine learning-based spam email classifier built using Python and Streamlit. It classifies emails as **spam** or **ham** (not spam) 
and allows users to provide feedback to improve the model.

## Features
- Text input and file upload options for email classification.
- User feedback system to correct predictions.
- Retraining pipeline to update the model with new data.
- Interactive web interface built with Streamlit.

## Technologies Used
- Python
- Streamlit (for the web app)
- Scikit-learn (for the machine learning model)
- NLTK (for text preprocessing)
- Joblib (for saving and loading models)

## How to Run
1. Clone this repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Access the app in your browser at `http://localhost:8501`.


## Custom Theme
The app uses a custom theme defined in `.streamlit/config.toml`. To use the same theme, ensure the `config.toml` file is present in the `.streamlit` folder.

![image](https://github.com/user-attachments/assets/019f7b6b-9e6b-461e-aa8f-352a70e3cb87)
