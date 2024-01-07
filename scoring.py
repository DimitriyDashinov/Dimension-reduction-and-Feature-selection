from data_loader import DataLoader
from ml_helper import DATA_PATH, FEATS_LIST, MODEL_NAME
import pandas as pd
import pickle

class ModelScoring:
    def __init__(self, model_name, df, feats):
        self.model_name = model_name
        self.df = df
        self.feats = feats

    def get_model(self):
        with open(self.model_name, 'rb') as file:  
            model = pickle.load(file)
        return model
    
    def score_with_model(self, model):
        df = self.df
        df = df[self.feats]
        pred = model.predict_proba(df)
        return pred

dl = DataLoader()
df = dl.load_clean_data(path=DATA_PATH)

ms = ModelScoring(MODEL_NAME, df, FEATS_LIST)

model = ms.get_model()
predictions = ms.score_with_model(model=model)
predictions = pd.DataFrame(predictions)
print(predictions.head)
