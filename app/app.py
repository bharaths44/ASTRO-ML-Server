# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import logging
import uvicorn
import io
from prophet import Prophet

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
    store_num: int = 0,
    item_num: int = 0,
    period_type: str = "M",
    num_periods: int = 3,
):
    """
    Handles HTTP POST requests to the '/predict' endpoint.

    Args:
        file (UploadFile): The uploaded CSV file containing the data.
        store_num (int): The store number to filter the data.
        item_num (int): The item number to filter the data.
        period_type (str): The type of period for the forecast ('M' for months, 'D' for days).
        num_periods (int): The number of periods to forecast.

    Returns:
        JSONResponse: A JSON response containing the forecast data.
    """
    if period_type == "M":
        period = num_periods * 31
    else:
        period = num_periods

    contents = await file.read()
    df_data = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df_data["date"] = pd.to_datetime(df_data["date"])
    df_data = df_data.sort_values(by="date")
    df_data = df_data[df_data["store"] == store_num]
    df_data = df_data[df_data["item"] == item_num]

    df_data = df_data.rename(columns={"date": "ds", "sales": "y"})
    
    model_prophet = Prophet()
    model_prophet.fit(df_data)
    
    df_future = model_prophet.make_future_dataframe(
        periods=period, freq="D", include_history=False
    )
    forecast_prophet = model_prophet.predict(df_future)
    forecast_prophet.round()
    
    forecast_json = forecast_prophet.to_json(orient="records", date_format="iso")
    return JSONResponse(forecast_json)


if __name__ == "__main__":
    """
    Entry point for running the FastAPI application with Uvicorn.
    """
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")