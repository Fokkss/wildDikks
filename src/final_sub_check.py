import pandas as pd


def save_final_submission(predictions, filename="output/prediction.csv"):
    df = pd.Series(predictions)

    if df.isnull().values.any():
        print("NaN detected!!!! :000, turning in 0.0")
        df = df.fillna(0.0)

    df.to_csv(filename, index=False, header=None, decimal=".", sep=",")

    print(f"file {filename} successfully saved")


# test subject
if __name__ == "__main__":
    test_data = [10.5, 20.3, 15.0, 90.09]
    save_final_submission(test_data)
