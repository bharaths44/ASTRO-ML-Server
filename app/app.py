# app.py

import io
import random
import logging
import uvicorn
import tempfile
import pandas as pd
import plotly.io as pio
from prophet import Prophet
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, File, HTTPException, UploadFile, Form

random.seed(1234)

app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@app.get("/")
def root():
    """
    Handles HTTP GET requests to the root endpoint ('/').

    Returns:
        dict: A JSON response containing a welcome message.
    """
    return {"message": "Welcome to the FastAPI!"}


@app.get("/info")
def info():
    """
    Handles HTTP GET requests to the '/info' endpoint.

    Returns:
        JSONResponse: A JSON response containing information about the API.
    """
    return JSONResponse(
        {
            "name": "FastAPI",
            "version": "1.0",
            "description": "This is a FastAPI for predictions.",
        }
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    store_num: Optional[int] = Form(None),
    item_num: Optional[int] = Form(None),
    period_type: str = Form(...),
    num_periods: int = Form(...),
):
    """
    Handles HTTP POST requests to the '/predict' endpoint.

    Args:
        file (UploadFile): The uploaded CSV file containing the data.
        store_num (Optional[int]): The store number to filter the data. If None, use all stores.
        item_num (Optional[int]): The item number to filter the data. If None, use all items.
        period_type (str): The type of period for the forecast ('M' for months, 'D' for days).
        num_periods (int): The number of periods to forecast.

    Returns:
        JSONResponse: A JSON response containing the forecast data.
    """
    try:
        logging.info(
            "Received request with period_type: %s, num_periods: %d",
            period_type,
            num_periods,
        )

        logging.info("Store number: %s", store_num)
        logging.info("Item number: %s", item_num)
        logging.info("Period type: %s", period_type)
        logging.info("Number of periods: %d", num_periods)

        if period_type == "M":
            period = num_periods * 31
        else:
            period = num_periods

        contents = await file.read()
        logging.info("File content length: %d", len(contents))

        df_data = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        logging.info("Dataframe shape: %s", df_data.shape)
        logging.info("Dataframe columns: %s", df_data.columns.tolist())

        # Strip whitespace from column names
        df_data.columns = df_data.columns.str.strip()
        logging.info("Stripped Dataframe columns: %s",
                     df_data.columns.tolist())

        # Filter by store_num and item_num if provided
        if store_num is not None:
            df_data = df_data[df_data["store"] == store_num]
        if item_num is not None:
            df_data = df_data[df_data["item"] == item_num]

        if df_data.empty:
            logging.error("No data found for the specified store and item.")
            raise HTTPException(
                status_code=404,
                detail="No data found for the specified store and item.",
            )

        if df_data.shape[0] < 2:
            logging.error("Dataframe has less than 2 non-NaN rows.")
            raise HTTPException(
                status_code=400,
                detail="Dataframe has less than 2 non-NaN rows.",
            )

        df_data = df_data.rename(columns={"date": "ds", "sales": "y"})
        df_data = df_data[["ds", "y"]]

        model_prophet = Prophet()
        model_prophet.fit(df_data)

        df_future = model_prophet.make_future_dataframe(
            periods=period, freq="D", include_history=False
        )

        forecast_prophet = model_prophet.predict(df_future)
        forecast_prophet.round()

        # Calculate the rolling average for the actual sales data
        df_data["sales_trend"] = df_data["y"].rolling(
            window=30, min_periods=1).mean()

        forecast_prophet["yhat_trend"] = forecast_prophet["yhat"].rolling(
            window=30, min_periods=1
        ).mean()

        forecast_prophet["yhat_upper_trend"] = forecast_prophet["yhat_upper"].rolling(
            window=30, min_periods=1
        ).mean()

        forecast_prophet["yhat_lower_trend"] = forecast_prophet["yhat_lower"].rolling(
            window=30, min_periods=1
        ).mean()

        # Create a subplot
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        fig.add_trace(
            go.Scatter(
                x=df_data.ds,
                y=df_data.sales_trend,
                name="Sales",
                mode="lines",
                showlegend=False  # Remove legend for this trace
            )
        )

        # Add the forecasted sales data
        fig.add_trace(
            go.Scatter(
                x=forecast_prophet.ds,
                y=forecast_prophet.yhat_trend,
                name="Forecast",
                mode="lines",
                showlegend=False  # Remove legend for this trace
            )
        )
        # Add the upper bound of the forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_prophet.ds,
                y=forecast_prophet.yhat_upper_trend,
                name="Upper Bound",
                mode="lines",
                line=dict(width=0),
                showlegend=False  # Remove legend for this trace
            )
        )

        # Add the lower bound of the forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_prophet.ds,
                y=forecast_prophet.yhat_lower_trend,
                name="Lower Bound",
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                showlegend=False  # Remove legend for this trace
            )
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmpfile:
            pio.write_image(fig, file=tmpfile.name,
                            format="svg", engine="kaleido")
            tmpfile_path = tmpfile.name

        # Return the SVG content as a response
        return FileResponse(tmpfile_path, media_type="image/svg+xml", filename="forecast_plot.svg")

    except pd.errors.EmptyDataError:
        logging.error("Uploaded file is empty or invalid.")
        raise HTTPException(
            status_code=400, detail="Uploaded file is empty or invalid."
        )
    except ValueError as ve:
        logging.error("Value error: %s", ve)
        raise HTTPException(status_code=400, detail=f"Value error: {ve}")
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


if __name__ == "__main__":
    """
    Entry point for running the FastAPI application with Uvicorn.
    """
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
