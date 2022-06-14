import os

"""
Use the below code to get the PWD and change the path accordingly or Use absolute path for the input

pwd = os.getcwd()
files = os.listdir(pwd)

dirname = os.path.dirname(__file__)
"""

# Taking the absolute path of files
dirname = os.getcwd()
parent_dir = dirname.split("src")[0]

TRAINING_FILE = os.path.join(parent_dir, 'input/input_kfold.csv')

TESTING_FILE = os.path.join(parent_dir, 'input/test.csv')

MODEL_OUTPUT = os.path.join(parent_dir, "models")
