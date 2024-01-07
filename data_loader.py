import pandas as pd
from sklearn.impute import KNNImputer

class DataLoader():
    '''Load and impute NaNs in dataset'''
    def __int__(self, path):
        self.path = path

    def load_clean_data(self, path):
        '''Load and impute NaNs in dataset'''
        df = pd.read_csv(path)
        df = df.drop('Time', axis=1)
        df['Pass/Fail'] = df['Pass/Fail'].replace(to_replace=[-1,1], value=[1,0])
        imputer = KNNImputer()
        imputer.fit(df)
        df = pd.DataFrame(imputer.transform(df), columns=df.columns)
        return df

