from flask import Flask, request, render_template
from joblib import load
import os

app = Flask(__name__)

# Load the trained model and vectorizer
# Load the vectorizer and model in Flask app
vectorizer = load("saved_data_models/vectorizer.pkl")
model = load("saved_data_models/Decision_Tree_best_model.pkl")  # or the appropriate model file name


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from form
    input_text = request.form['input_text']
    
    # Transform input text with vectorizer
    input_vec = vectorizer.transform([input_text])
    
    # Predict emotion
    predicted_emotion = model.predict(input_vec)[0]
    
    return render_template('index.html', prediction=predicted_emotion)

if __name__ == "__main__":
    app.run(debug=True)
