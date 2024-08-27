import pandas as pd
import logging

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def graph_data(output, predictions, store_num, item_num, data):
    fc = forecast(output, predictions)
    res = predicted_graph(store_num, item_num, train=data, forecast=fc)
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


def predicted_graph(store_num, item_num, train, forecast):
    logging.info(f"Preparing predicted data for store {store_num}, item {item_num}...")

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
    logging.info(f"Plotting predicted graph for store {store_num}, item {item_num}...")
  

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    fig.add_trace(
        go.Scatter(
            x=train_data.date,
            y=train_data.sales,
            name=f"Store {store_num} Item {item_num} Sales",
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_data.date,
            y=forecast_data.sales,
            name=f"Store {store_num} Item {item_num} Forecast",
            mode="lines",
        )
    )

    fig.update_layout(
        title=f"Sales and Forecast for Store {store_num}, Item {item_num}",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend_title="Legend",
        hovermode="x unified",
    )

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )

    fig.show()
    # Combine the dictionaries into a single JSON response
    response = {"train": train_dict, "forecast": forecast_dict}

    logging.info(f"Predicted data for store {store_num}, item {item_num} prepared.")

    return response
