from flask import Flask, jsonify, request
import pandas as pd
import random
from flask_cors import CORS
import copy
from pipeline import GetPrediction
app = Flask(__name__)
CORS(app)

prediction_logic = GetPrediction()


@app.route('/model', methods=['POST'])
def predict():
    # data = df.sample(int(index))
    data = request.json
    # data = pd.DataFrame(data)
    # print(data)

    df_data = data[0]
    for key,val in df_data.items():
        df_data[key] = [val]
    for data_point in data[1:]:
        for key,val in data_point.items():
            df_data[key].append(val)

    data = pd.DataFrame(df_data)
    results = prediction_logic.get_predictions(data)

    return pd.DataFrame({"predictions":results}).to_json()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
