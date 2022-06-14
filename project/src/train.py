# Importing necessary packages


import os
import config
import model_dispatcher

import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn import metrics


def run(fold, ml_model):
    # Reading the input data
    df = pd.read_csv(config.TRAINING_FILE)

    # fold parameter is for test model. So we are selecting all folds expect "fold" for training
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where the kfold = fold(parameter)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Creating X_train, y_train - Converting all data to numpy array
    X_train = df_train.drop(["label", "kfold"], axis=1).values
    y_train = df_train.label.values

    # Creating X_valid, y_valid - Converting all data to numpy array
    X_valid = df_valid.drop(["label", "kfold"], axis=1).values
    y_valid = df_valid.label.values

    # Initializing Simple Decision Tree Classifier
    model = model_dispatcher.models[ml_model]

    # Fitting the model with train data
    model.fit(X_train, y_train)

    # Predicting for validation data
    pred = model.predict(X_valid)

    # Calculating accuracy -- as the data is even(not skewed)
    accuracy = metrics.accuracy_score(y_valid, pred)
    print(f"Fold = {fold}, Accuracy = {accuracy}, model = {ml_model}")  

    # Save the model
    model_output = os.path.join(config.MODEL_OUTPUT,f"dt_{fold}.bin")
    joblib.dump(model, model_output)


if __name__ == "__main__":

    # Initialize the ArgumentParser from argparse
    parser = argparse.ArgumentParser()

    # Adding the required arguments and their types
    parser.add_argument("-fold", type=int)
    parser.add_argument("-model", type=str)

    # Reading the arguments from the command line
    args = parser.parse_args()

    # Run the fold from argument(input)
    run(fold = args.fold, ml_model= args.model)

"""
    predicting with various folds(for validation test)
    for i in range(0,10):   
        run(i)

    run(7)
    run(5)
"""