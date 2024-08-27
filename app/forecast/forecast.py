import pandas as pd
import logging


def graph_data(output, predictions, store_num, item_num, data):
    fc = forecast(output, predictions)
    res = prepare_data(store_num, item_num, train=data, forecast=fc)
    return res


def forecast(output, predictions):
    logging.info("Creating forecast DataFrame...")

    # Log lengths of the arrays
    logging.info(f"Length of output['date']: {len(output['date'])}")
    logging.info(f"Length of output['store']: {len(output['store'])}")
    logging.info(f"Length of output['item']: {len(output['item'])}")
    logging.info(f"Length of predictions: {len(predictions)}")

    # Ensure all arrays have the same length
    min_length = min(
        len(output["date"]), len(output["store"]), len(output["item"]), len(predictions)
    )
    logging.info(f"Minimum length: {min_length}")

    forecast = pd.DataFrame(
        {
            "date": output["date"][:min_length],
            "store": output["store"][:min_length],
            "item": output["item"][:min_length],
            "sales": predictions[:min_length],
        }
    )
    logging.info("Forecast created.")
    return forecast


def prepare_data(store_num, item_num, train, forecast):
    logging.info(f"Preparing data for store {store_num}, item {item_num}...")

    # Filter the data
    train_data = train[(train.store == store_num) & (train.item == item_num)]
    forecast_data = forecast[
        (forecast.store == store_num) & (forecast.item == item_num)
    ]

    # Convert the data to dictionaries
    train_dict = {
        "date": train_data.date.tolist(),
        "sales": train_data.sales.tolist(),
        "label": f"Store {store_num} Item {item_num} Sales",
    }

    forecast_dict = {
        "date": forecast_data.date.tolist(),
        "sales": forecast_data.sales.tolist(),
        "label": f"Store {store_num} Item {item_num} Forecast",
    }

    # Combine the dictionaries into a single JSON response
    response = {"train": train_dict, "forecast": forecast_dict}

    logging.info(f"Data for store {store_num}, item {item_num} prepared.")

    return response
