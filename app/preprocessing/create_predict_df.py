import pandas as pd
import itertools
import logging


def create_predict_df(train_data, period_type="D", num_periods=365):
    """
    Create a DataFrame for prediction with a specified period type and number of periods.

    Parameters:
    - train_data: DataFrame, the training data.
    - period_type: str, type of period ('D' for days, 'M' for months, 'Y' for years).
    - num_periods: int, number of periods to generate.

    Returns:
    - DataFrame, the prediction data.
    """
    # Load the training data to get the last date
    train_data["date"] = pd.to_datetime(train_data["date"])
    last_date = train_data["date"].max()
    logging.info(f"Last date in training data: {last_date}")

    # Generate a date range based on the specified period type and number of periods
    if period_type == "Y":
        date_range = pd.date_range(
            start=last_date + pd.DateOffset(days=1), periods=num_periods * 365, freq="D"
        )
    elif period_type == "M":
        date_range = pd.date_range(
            start=last_date + pd.DateOffset(days=1), periods=num_periods * 30, freq="D"
        )
    else:
        date_range = pd.date_range(
            start=last_date + pd.DateOffset(days=1),
            periods=num_periods,
            freq=period_type,
        )

    # Get unique stores and items
    stores = train_data["store"].unique()
    items = train_data["item"].unique()

    # Create combinations of stores and items
    combinations = list(itertools.product(stores, items))

    # Create a DataFrame with the combinations and date range
    predict_data = pd.DataFrame(combinations, columns=["store", "item"])
    predict_data = (
        predict_data.assign(key=1)
        .merge(pd.DataFrame(date_range, columns=["date"]).assign(key=1), on="key")
        .drop("key", axis=1)
    )
    predict_data = predict_data.sort_values(by=["store", "item", "date"])

    logging.info("Predict DataFrame created")
    return predict_data
