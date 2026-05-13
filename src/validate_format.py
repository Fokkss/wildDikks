import pandas as pd
import sys


def validate_submission(file_path, expected_rows=None):
    try:
        df = pd.read_csv(file_path, header=None)

        if df.shape[1] != 1:
            print(f"error: number of columns doesn't match: {df.shape[1]}")
            sys.exit(1)

        # if expected_rows and len(df) != expected_rows:
        #     print(f"error: expected number of rows: {expected_rows}, found: {len(df)}")
        #     sys.exit(1)

        if not pd.api.types.is_numeric_dtype(df[0]):
            print("error: invalid type of cells")
            sys.exit(1)

        if df[0].isnull().any():
            print("error: NaN values found")
            sys.exit(1)

        print("success!")

    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    validate_submission("output/prediction.csv")
