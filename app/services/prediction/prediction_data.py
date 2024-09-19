
import logging
import pandas as pd
from prophet import Prophet
from typing import Optional
from .utils import process_file
from fastapi.responses import JSONResponse
from fastapi import HTTPException, UploadFile, File, Form


async def predict_data(
    file: UploadFile = File(...),
    store_num: Optional[int] = Form(None),
    item_num: Optional[int] = Form(None),
    period_type: str = Form(...),
    num_periods: int = Form(...),
):
    try:
        df_data, period = await process_file(file, store_num, item_num, period_type, num_periods)

        # Ensure 'ds' column is datetime
        df_data['ds'] = pd.to_datetime(df_data['ds'], errors='coerce')
        if df_data['ds'].isnull().any():
            raise ValueError("Invalid date format in 'ds' column")

        model_prophet = Prophet()
        model_prophet.fit(df_data,seed=4)

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

        # Ensure 'ds' column in forecast is datetime
        forecast_prophet['ds'] = pd.to_datetime(
            forecast_prophet['ds'], errors='coerce')
        if forecast_prophet['ds'].isnull().any():
            raise ValueError("Invalid date format in 'ds' column of forecast")

        # Convert 'ds' column to string format
        df_data["ds"] = df_data["ds"].dt.strftime('%Y-%m-%d %H:%M:%S')
        forecast_prophet["ds"] = forecast_prophet["ds"].dt.strftime(
            '%Y-%m-%d %H:%M:%S')

        # Prepare the data to be returned as JSON
        response_data = {
            "actual_sales": df_data[["ds", "sales_trend"]].to_dict(orient="records"),
            "forecast": forecast_prophet[["ds", "yhat_trend", "yhat_upper_trend", "yhat_lower_trend"]].to_dict(orient="records"),
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
