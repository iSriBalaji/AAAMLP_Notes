# Importing necessary packages

import os
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree

"""
Use the below code to get the PWD and change the path accordingly or Use absolute path for the input

pwd = os.getcwd()
files = os.listdir(cwd)
"""


def run(folds):
    pass
    # Reading the input data
    df = pd.read_csv("input/train.csv")
    print(df.head(5))

run(5)
