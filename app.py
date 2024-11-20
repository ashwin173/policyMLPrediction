from flask import Flask, request, render_template
import numpy as np
import pickle

app= Flask('__name__')

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Define default values for all features
DEFAULT_VALUES = {
    'age': 35, 'call_day': 15, 'call_duration': 300, 'call_frequency': 2, 
    'conversion_status': 0, 
    'occupation_administrative_staff': 0, 'occupation_business_owner': 0, 
    'occupation_domestic_worker': 0, 'occupation_executive': 0, 
    'occupation_independent_worker': 0, 'occupation_jobless': 0, 
    'occupation_manual_worker': 0, 'occupation_retired_worker': 0, 
    'occupation_service_worker': 0, 'occupation_student': 0, 
    'occupation_technical_specialist': 0, 'occupation_unidentified': 0, 
    'education_level_college': 1, 'education_level_elementary_school': 0, 
    'education_level_high_school': 0, 'education_level_unidentified': 0, 
    'marital_status_divorced': 0, 'marital_status_married': 1, 
    'marital_status_single': 0, 
    'communication_channel_landline': 0, 'communication_channel_mobile': 1, 
    'communication_channel_unidentified': 0, 
    'call_month_January': 0, 'call_month_February': 0, 
    'call_month_March': 0, 'call_month_April': 0, 
    'call_month_May': 0, 'call_month_June': 0, 
    'call_month_July': 0, 'call_month_August': 0, 
    'call_month_September': 0, 'call_month_October': 0, 
    'call_month_November': 0, 'call_month_December': 0
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Initialize feature array with default values
    feature_array = np.array(list(DEFAULT_VALUES.values()))
    
    # Update with user-provided values
    feature_array[0] = int(request.form.get('age', DEFAULT_VALUES['age']))
    feature_array[1] = int(request.form.get('call_day', DEFAULT_VALUES['call_day']))
    
    # Scale call_duration to give it higher importance
    call_duration = int(request.form.get('call_duration', DEFAULT_VALUES['call_duration']))
    feature_array[2] = call_duration * 3  # Boosting importance of call_duration
    
    feature_array[3] = int(request.form.get('call_frequency', DEFAULT_VALUES['call_frequency']))
    
    # Handle occupation
    occupation = request.form.get('occupation', 'occupation_unidentified')
    feature_array[list(DEFAULT_VALUES.keys()).index(occupation)] = 1

    # Handle call month
    call_month = request.form.get('call_month', 'call_month_January')
    feature_array[list(DEFAULT_VALUES.keys()).index(call_month)] = 1
    
    # Debugging: Check feature array before prediction
    print("Feature Array Before Prediction:", feature_array)

    # Make predictions
    prediction = model.predict([feature_array])
    prediction_mapping = {0: "Not Interested", 1: "Maybe Interested", 2: "Highly Interested"}
    result = prediction_mapping[prediction[0]]

    if call_duration<10:
        result="Not Interested "
    elif 10<=call_duration<50:
        result="Maybe Intersted"
    else:
        result="Hightly Intersted"
         
    return f"<h1>Prediction: {result}</h1>"



if __name__ == '__main__':
    app.run(debug=True)
