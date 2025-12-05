import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

class FeaturesEngineering:
    def split_train_test():
        data = pd.read_csv('./data/raw.csv')
        X = data.drop('Close', axis=1)
        y = data['Close']

        null_values = data.isnull()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return (X_train, X_test, y_train, y_test)
