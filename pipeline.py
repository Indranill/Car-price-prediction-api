from Inference import Inference_Pipeline
import pickle5 as pickle
import numpy as np
import pandas as pd


class GetPrediction:
    def __init__(self):
        self.inference = Inference_Pipeline()

        with open('Artifacts/scalar.pkl', 'rb') as handle:
            p = pickle.Unpickler(handle)
            self.scalar = p.load()

        with open('Artifacts/encoder.pkl', 'rb') as handle:
            p = pickle.Unpickler(handle)
            self.encoder = p.load()
            
        with open('Artifacts/Model.pkl', 'rb') as handle:
            p = pickle.Unpickler(handle)
            self.model = p.load()
    
    
    def get_predictions(self, data):
        print(self.encoder)
        cleaned_data = self.inference.inference_pipeline(data,
                                                         self.encoder, 
                                                         self.scalar)
        return self.model.predict(cleaned_data)


# test = pd.read_csv("Notebooks/test.csv")

# inference = Inference_Pipeline()

# with open('Artifacts/scalar.pkl', 'rb') as handle:
#     p = pickle.Unpickler(handle)
#     scalar = p.load()

# with open('Artifacts/encoder.pkl', 'rb') as handle:
#     p = pickle.Unpickler(handle)
#     encoder = p.load()
    
# with open('Artifacts/Model.pkl', 'rb') as handle:
#     p = pickle.Unpickler(handle)
#     model = p.load()
    

    
# test = pd.read_csv("./preds.csv")
# print(test.head())
# test = inference.car_data_inference(test,standard_scalar, dict_encoder, car_list)
# print(np.exp(model.predict(test)))

# def get_predictions(test_data):
#     try:
#         cleaned_data =  inference.inference_pipeline(data, encoder, scalar)
#         return np.exp(model.predict(cleaned_data))
#     except Exception as e:
#         return None

