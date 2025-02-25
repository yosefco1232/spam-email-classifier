@echo off

:: Activate virtual environment
call venv\Scripts\activate

:: Run the retraining script
python retrain_model.py

:: Move the new model files to the app directory
for %%f in (spam_classifier_*.pkl) do (
    move /Y "%%f" spam_classifier.pkl
)
for %%f in (tfidf_vectorizer_*.pkl) do (
    move /Y "%%f" tfidf_vectorizer.pkl
)

echo Retraining complete and model files updated.
pause