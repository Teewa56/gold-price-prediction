from sklearn.linear_model import LinearRegression
from data_preprocessing.features_engineering import FeaturesEngineering
from sklearn.metrics import r2_score, mean_square_error
import matplotlib as plt
import seaborn
import joblib

class LinearModel:
    def __init__(self, LinearRegression, FeaturesEngineering):
        self.model = LinearRegression
        self.features_engineering = FeaturesEngineering

    def train_model():
        X_train, X_test, y_train, y_test = self.features_engineering.split_train_test

        linear_model = self.model

        linear_model.fit(X_train, y_train)#train with the training data

        train_prediction = linear_model.predcit(X_train)#predict with the training data

        #training evaluation
        r2 = r2_score(X_train, train_prediction)
        mse = mean_square_error(X_train, train_prediction)
        
        print("mse : {}, r2 : {}", mse, r2)

        test_prediction = linear_model.predcit(X_test)#predict with the training data

        #test evaluation
        r2 = r2_score(X_test, test_prediction)
        mse = mean_square_error(X_test, test_prediction)

        print("mse : {}, r2 : {}", mse, r2)


def main():
    train_model = LinearModel.train_model()
