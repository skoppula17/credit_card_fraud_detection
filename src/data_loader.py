import pandas as pd

class DataLoader:
    """
    Loads the credit card dataset and provides summary methods.
    """
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def summary(self, df: pd.DataFrame) -> None:
        print("\n--- Head ---")
        print(df.head())
        print("\n--- Info ---")
        print(df.info())
        print("\n--- Describe ---")
        print(df.describe())