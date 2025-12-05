from sklearn.linear_model import LinearRegression
from data_preprocessing.features_engineering import FeaturesEngineering
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

def train_and_evaluate_model():
    ( X_train, X_test, y_train, y_test ) = FeaturesEngineering.split_train_test()
    
    linear_model = LinearRegression()

    linear_model.fit(X_train, y_train)  # train with the training data

    train_prediction = linear_model.predict(X_train)  # predict with the training data
    print(train_prediction)
    # training evaluation
    r2 = r2_score(y_train, train_prediction)
    mse = mean_squared_error(y_train, train_prediction)

    print(f"mse : {mse}, r2 : {r2}")

    test_prediction = linear_model.predict(X_test)  # predict with the test data

    # test evaluation
    r2 = r2_score(y_test, test_prediction)
    mse = mean_squared_error(y_test, test_prediction)

    print(f"mse : {mse}, r2 : {r2}")

    joblib.dump(linear_model, "models/linear_model.pkl")


if __name__ == "__main__":
    train_model()