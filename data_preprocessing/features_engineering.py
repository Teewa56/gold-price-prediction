import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

class FeaturesEngineering:
    data = pd.read_csv('./data/raw.csv')

    print(data.head())

    X = data.drop('Close', axis=1)
    y = data['Close']

    print(X.head())
    print(y.head())

    null_values = data.isnull()

    print(null_values)
    print(null_values.sum())

    def split_train_test():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test
