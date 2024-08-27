import logging
import numpy as np

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe)))


def roll_mean_features(dataframe, windows):
    logging.info("Creating rolling mean features...")
    for window in windows:
        dataframe["sales_roll_mean_" + str(window)] = dataframe.groupby(
            ["store", "item"]
        )["sales"].transform(
            lambda x: x.shift(1)
            .rolling(window=window, min_periods=10, win_type="triang")
            .mean()
        ) + random_noise(
            dataframe
        )
    logging.info("Rolling mean features created.")
    return dataframe
