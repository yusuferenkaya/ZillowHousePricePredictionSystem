import pickle

import numpy as np
import xgboost
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Get the data from the form
        property_type = request.form.get('property_type')
        # state = request.form.get('location')
        # city
        bedrooms = int(request.form.get('bedrooms'))
        bathrooms = int(request.form.get('bathrooms'))
        land_space = float(request.form.get('land_space'))
        living_space = float(request.form.get('living_space'))

        state = request.form.get('state')
        city = request.form.get('city')
        postal_code = request.form.get('postal_code')









        # Process the data...
        result = process_data(postal_code, property_type, state, bedrooms, bathrooms, city, land_space, living_space)
        prediction = predict_the_price(postal_code, property_type, state, bedrooms, bathrooms, city, land_space, living_space)
        # Render the result page
        return render_template("result.html", result=result, prediction=prediction)

    return render_template("form.html",states=states,property_types=property_types)

def process_data(postal_code, property_type, state, bedrooms, bathrooms, city, land_space, living_space):
    # Process the data and return the result
    result = {
        'property_type': property_type,
        'city': city,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'living_space': living_space,
        'postal_code': postal_code,
        'state': state,
        'land_space': land_space
    }
    return result

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/cities/<state>')
def get_cities(state):
    # Get the list of cities for the selected state
    cities = list(set(df[df["state"] == state]["city"]))
    cities.sort()

    # Return the list of cities as a JSON response
    return {'cities': cities}


@app.route('/postal_codes/<city>/<state>')
def get_postal_codes(city, state):
    # Get the list of postal codes for the selected city from the postal_codes_by_city dictionary
    postal_codes = list(set(df[(df["state"] == state) & (df["city"] == city)]["postcode"]))
    postal_codes.sort()
    # Return the list of postal codes as a JSON response
    return {'postal_codes': postal_codes}

def predict_the_price(postal_code, property_type, state, bedrooms, bathrooms, city, land_space, living_space):
    with open('house_model_prediction.pkl', 'rb') as f:
        machine_model = pickle.load(f)
    with open('le_for_state.pkl', 'rb') as statele:
        le_for_state = pickle.load(statele)
    with open('le_for_city.pkl', 'rb') as cityle:
        le_for_city = pickle.load(cityle)
    with open('le_for_postcode.pkl', 'rb') as postcodele:
        le_for_postcode = pickle.load(postcodele)
    with open('le_for_property_type.pkl', 'rb') as propertyle:
        le_for_property_type = pickle.load(propertyle)
    with open('minmax_scaler_for_living_space.pkl', 'rb') as minmax:
        # Load the scaler object from the file
        minmax_scaler_for_living_space = pickle.load(minmax)

    # Deserialize the label encoder object
    input_data = {'city': [city], 'state': [state], 'postcode': [postal_code], 'bedroom_number': [bedrooms],
                  'bathroom_number': [bathrooms], 'living_space': [living_space], 'land_space': [land_space],
                  'property_type': [property_type]}
    input_data = pd.DataFrame(input_data)

    living_space_array = np.array(living_space)
    living_space_array = living_space_array.reshape(1, -1)
    input_data['living_space'] = minmax_scaler_for_living_space.transform(living_space_array)
    input_data['city'] = le_for_city.transform(input_data['city'])
    input_data['state'] = le_for_state.transform(input_data['state'])
    input_data['postcode'] = le_for_postcode.transform(input_data['postcode'])
    input_data['property_type'] = le_for_property_type.transform(input_data['property_type'])
    pred = machine_model.predict(input_data)
    if pred < 0:
        pred = pred * -1
    pred = int(pred)
    return pred
df = pd.read_csv("data_saved.csv",low_memory=False)


property_types = list(set(df['property_type'].tolist()))
states = list(set(df["state"]))
states.sort()
