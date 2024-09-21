import logging
from typing import Optional

import pandas as pd
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from prophet import Prophet

from .utils import process_file


async def predict_data(
    file: UploadFile = File(...),
    store_num: Optional[int] = Form(None),
    item_num: Optional[int] = Form(None),
):
    try:
        # Process the file to get the data
        df_data = await process_file(file, store_num, item_num)

        # Ensure 'ds' column is datetime
        df_data["ds"] = pd.to_datetime(df_data["ds"], errors="coerce")
        if df_data["ds"].isnull().any():
            raise ValueError("Invalid date format in 'ds' column")

        # Determine period_type and num_periods
        data_length = len(df_data)
        date_diff = (df_data["ds"].max() - df_data["ds"].min()).days

        if date_diff > 365:
            num_periods = min(int(data_length * (60 / 365)), data_length)
        elif date_diff > 30:
            num_periods = min(30, data_length)
        else:
            num_periods = data_length

        # Log the determined period_type and num_periods
        logging.info("Determined num_periods: %d", num_periods)

        # Fit the Prophet model
        model_prophet = Prophet()
        model_prophet.fit(df_data, seed=4)

        # Create future dataframe
        df_future = model_prophet.make_future_dataframe(
            periods=num_periods, freq="D", include_history=False
        )

        # Ensure future dataframe is created
        if df_future.empty:
            raise ValueError("Future dataframe is empty")

        # Predict future values
        forecast_prophet = model_prophet.predict(df_future)
        forecast_prophet.round()

        df_data["sales_trend"] = df_data["y"].rolling(window=30, min_periods=1).mean()

        forecast_prophet["yhat_trend"] = (
            forecast_prophet["yhat"].rolling(window=30, min_periods=1).mean()
        )

        forecast_prophet["yhat_upper_trend"] = (
            forecast_prophet["yhat_upper"].rolling(window=30, min_periods=1).mean()
        )

        forecast_prophet["yhat_lower_trend"] = (
            forecast_prophet["yhat_lower"].rolling(window=30, min_periods=1).mean()
        )

        # Ensure 'ds' column in forecast is datetime
        forecast_prophet["ds"] = pd.to_datetime(forecast_prophet["ds"], errors="coerce")
        if forecast_prophet["ds"].isnull().any():
            raise ValueError("Invalid date format in 'ds' column of forecast")

        # Convert 'ds' column to string format
        df_data["ds"] = df_data["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")
        forecast_prophet["ds"] = forecast_prophet["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the data to be returned as JSON
        response_data = {
            "actual_sales": df_data[["ds", "sales_trend"]].to_dict(orient="records"),
            "forecast": forecast_prophet[
                ["ds", "yhat_trend", "yhat_upper_trend", "yhat_lower_trend"]
            ].to_dict(orient="records"),
        }

        return JSONResponse(content=response_data)

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
