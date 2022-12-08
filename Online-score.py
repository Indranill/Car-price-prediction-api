import os
import logging
import json
import numpy
import joblib


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path =  Model.get_model_path('car-price-model')

    encoder_path = Model.get_model_path('label-encoder')

    scalar_path = Model.get_model_path('standard-scalar')

    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    scalar = joblib.load(scalar_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)
    # df_data = data
    # for key,val in df_data.items():
    #     df_data[key] = [val]
    # for data_point in data[1:]:
    #     for key,val in data_point.items():
    #         df_data[key].append(val)

    # data = pd.DataFrame(df_data)
    # results = prediction_logic.get_predictions(data)
    return data