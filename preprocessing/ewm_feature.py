import logging


def ewm_features(dataframe, alphas, lags):
    logging.info("Creating exponentially weighted mean features...")
    for alpha in alphas:
        for lag in lags:
            dataframe[
                "sales_ewm_alpha_" + str(alpha).replace(".", "") + "_lag_" + str(lag)
            ] = dataframe.groupby(["store", "item"])["sales"].transform(
                lambda x: x.shift(lag).ewm(alpha=alpha).mean()
            )
    logging.info("Exponentially weighted mean features created.")
    return dataframe
