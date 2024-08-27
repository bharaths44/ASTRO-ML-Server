import numpy as np
import logging
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe)))


def lag_features(dataframe, lags):
    logging.info("Creating lag features...")
    for lag in lags:
        dataframe["sales_lag_" + str(lag)] = dataframe.groupby(["store", "item"])[
            "sales"
        ].transform(lambda x: x.shift(lag)) + random_noise(dataframe)
    logging.info("Lag features created.")
    return dataframe
