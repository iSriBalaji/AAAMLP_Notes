## This script will create a new csv file which is same as train.csv but the data is shuffles and has a column kfolds
## here, using kfold as all the classes are even; use stratified if the data is skewed
## pg no 23 - AAAMLP

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # Reading the input data
    df = pd.read_csv("input/input.csv")      # 60000 rows

    # Creating a new column kfolds and filling it with -1
    df["kfold"] = -1

    # Randomize the rows and resetting the index; frac = 1 (100% of data)
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)

    # Creating Kfold object from model_selection
    kf = model_selection.KFold(n_splits=7)

    # Fill the kfold column with corresponding k-th value
    for fold, (trn_, val_) in enumerate(kf.split(X = df)):
        df.loc[val_, 'kfold'] = fold

    # Saving the file as a separate csv file
    df.to_csv("input/input_kfold.csv", index=False)

    # print(df.head())
    print(df.shape)