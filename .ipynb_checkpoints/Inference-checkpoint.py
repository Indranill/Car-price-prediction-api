# load data
import pandas as pd
import numpy as np
# stats test
import scipy.stats as stat
# training and preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Inference_Pipeline:
    def handle_engine_volume(self, ele):
        val =  float(str(ele).split(" ")[0])
        return val if val!=np.nan else 0

    def handle_levy(self,x):
        if x=='-':
            return 0
        else:
            return float(x)

    def handle_mileage(self, x):
        return float(x.split(" ")[0])

    def inference_pipeline(self, train,label_encoders, scalers):
        if 'ID' in train.columns: 
            train = train.drop(["ID"], axis=1)
        if 'Price' in train.columns: 
            train = train.drop(["Price"], axis=1)

        train["Levy"] = train["Levy"].apply(self.handle_levy)
        train["Engine volume"] = train["Engine volume"].apply(self.handle_engine_volume)
        train["Mileage"] = train["Mileage"].apply(self.handle_mileage)
        
        numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [x for x in numeric_cols if x!="Price"]
        cat_cols = [x for x in train.columns if x not in numeric_cols and x!="Price"]

        print(numeric_cols)
        print(cat_cols)
        
        for i in cat_cols:
            if '<unknown>' not in label_encoders[i].classes_ :
                train[i] = train[i].map(lambda s: '<unknown>' if s not in label_encoders[i].classes_ else s)
                label_encoders[i].classes_ = np.append(label_encoders[i].classes_, '<unknown>')
            else:
                train[i] = train[i].map(lambda s: '<unknown>' if s not in label_encoders[i].classes_ else s)
        for col in cat_cols:
            train[col] = label_encoders[col].transform(train[col])

        for col in numeric_cols:
            train[col] = scalers[col].transform(train[col].values.reshape(-1, 1)).squeeze()
                                        
        return train
