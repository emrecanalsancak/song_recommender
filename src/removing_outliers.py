import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/processed/01_songs_no_missing.csv")
cols_with_outliers = ["loudness", "speechiness"]

# ------------ OUTLIERS ------------ #


def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def remove_outliers(dataframe, col_names, q1=0.05, q3=0.95):
    df_without_outliers = dataframe.copy()

    for col_name in col_names:
        low_limit, up_limit = outlier_thresholds(df_without_outliers, col_name, q1, q3)
        df_without_outliers = df_without_outliers[
            ~(
                (df_without_outliers[col_name] < low_limit)
                | (df_without_outliers[col_name] > up_limit)
            )
        ]

    return df_without_outliers


if __name__ == "__main__":
    print("Processing data...")
    no_outliers = remove_outliers(df, cols_with_outliers)
    no_outliers.to_csv("../data/processed/02_no_outliers.csv", index=False)
