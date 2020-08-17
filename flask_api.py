from flask import Flask
from flask import request
import pandas as pd
import pickle
import os
app = Flask(__name__)


with open('housing_rfr_model.pickle', 'rb') as f:
    model = pickle.load(f)

# http://localhost:5000/predict_single?MedInc=7.33&HouseAge=28&AveRooms=4.55&AveBedrms=2.41&Population=299&AveOccup=2.66&Latitude=37.81&Longitude=-122.28


@app.route('/predict_single')
def predict_single():
    med_inc = float(request.args.get('MedInc'))
    house_age = float(request.args.get('HouseAge'))
    avg_rooms = float(request.args.get('AveRooms'))
    avg_bedrooms = float(request.args.get('AveBedrms'))
    population = float(request.args.get('Population'))
    avg_occup = float(request.args.get('AveOccup'))
    latitude = float(request.args.get('Latitude'))
    longitude = float(request.args.get('Longitude'))

    prediction = model.predict([[med_inc, house_age, avg_rooms, avg_bedrooms, population, avg_occup, latitude, longitude]])

    return str(prediction[0])


@app.route('/predict', methods=["POST"])
def predict():
    req = request.get_json()
    df = pd.DataFrame.from_dict(req)
    prediction = model.predict(df)

    return str(prediction)


if __name__ == '__main__':
    # app.run()

    port = os.environ.get('PORT')

    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()
