import os
import sys
from flask import Flask, request, render_template, jsonify

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

application = Flask(__name__)
app = application

# Load model function

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        data = CustomData(
            gender = data['gender'],
            race_ethnicity = data['race_ethnicity'],
            parental_level_of_education = data['parental_level_of_education'],
            lunch = data['lunch'],
            test_preparation_course = data['test_preparation_course'],
            reading_score = data['reading_score'],
            writing_score = data['writing_score']
        )

        prediction = PredictPipeline().predict(data.get_data_as_DF())

        # Ensure the prediction is within 0-100 range
        prediction = max(0, min(100, prediction))
        
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400