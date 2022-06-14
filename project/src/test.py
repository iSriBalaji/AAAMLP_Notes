import joblib
import config
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def test(model_no):
    # Reading the input data
    df = pd.read_csv(config.TESTING_FILE)
    print(df.shape)

    model_path = os.path.join(config.MODEL_OUTPUT,f"dt_{model_no}.bin")
    loaded_model = joblib.load(model_path)

    predict = loaded_model.predict(df)

    output = pd.DataFrame(list(zip(list(df.index+1), predict.tolist())), columns=["ImageId", "Label"])
    print(output.head())

    output.to_csv("input/submission.csv", index=False)

if __name__ == "__main__":
    test(8)