# app.py

from typing import Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging
import uvicorn
import io

import random

random.seed(1234)

from prophet import Prophet

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://<your-app-name>.azurewebsites.net",
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    store_num: Optional[int] = None,
    item_num: Optional[int] = None,
    period_type: str = "M",
    num_periods: int = 3,
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
        if period_type == "M":
            period = num_periods * 31
        else:
            period = num_periods

        contents = await file.read()
        df_data = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        df_data["date"] = pd.to_datetime(df_data["date"])
        df_data = df_data.sort_values(by="date")

        # Check if the 'store' column exists
        if "store" in df_data.columns and store_num is not None:
            # Check if there is only one store
            if df_data["store"].nunique() == 1:
                store_num = df_data["store"].unique()[0]

            df_data = df_data[df_data["store"] == store_num]

        # Check if the 'item' column exists and filter by item_num if provided
        if "item" in df_data.columns and item_num is not None:
            df_data = df_data[df_data["item"] == item_num]

        if df_data.empty:
            raise HTTPException(
                status_code=404,
                detail="No data found for the specified store and item.",
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

        forecast_json = forecast_prophet.to_json(orient="records", date_format="iso")
        return JSONResponse(forecast_json)

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400, detail="Uploaded file is empty or invalid."
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value error: {ve}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


if __name__ == "__main__":
    """
    Entry point for running the FastAPI application with Uvicorn.
    """
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
