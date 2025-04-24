from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load CSV data
data = pd.read_csv("aakash data - Sheet3.csv")

y1 =np.array([data['cases'].copy()])
y_mean=np.mean(y1)
y_std=np.std(y1)
y_new=(y1-y_mean)/y_std

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
svr=pickle.load(open("svr.pkl", "rb"))

categorical_cols = ['state', 'mosquito']
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve form data
    form_data = request.form

    # Convert form data into pandas DataFrame
    input_data = {
        'rainfall in mm': [float(form_data['feature2'])],
        'temperature': [float(form_data['feature3'])],
        'avg relative humidity': [float(form_data['feature4'])],
        'state': [form_data['feature5']],
        'mosquito': [form_data['feature6']]
    }
    df = pd.DataFrame(input_data)

    # Preprocess input data
    numeric_features = ['rainfall in mm', 'temperature', 'avg relative humidity']
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    # Encode categorical features
    encoded_data = encoder.transform(df[categorical_cols])
    df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols)

    # Concatenate numeric and encoded categorical features
    df_combined = pd.concat([df[numeric_features], df_encoded], axis=1)

    # Make prediction
    no_of_cases=svr.predict(df_combined)
    no_of_cases=no_of_cases[0]
    no_of_cases=abs(int((no_of_cases*y_std)+y_mean))
    prediction = model.predict(df_combined)
    prediction=prediction[0]
    probability=model.predict_proba(df_combined)
    probability=np.max(probability)

    if no_of_cases>1500:
        cat="Category A"
    elif no_of_cases>800 and no_of_cases<1500:
        cat="Category B"
    elif no_of_cases>300 and no_of_cases<800:
        cat="Category C"
    else:
        cat="Category D"

    return render_template("index.html", prediction_text='The prediction is {}'.format(prediction),pt2="The probability is {}".format(probability),pt3="Number of cases is {}".format(no_of_cases),pt4="Category is {}".format(cat))

if __name__ == "__main__":
    app.run(debug=True)
