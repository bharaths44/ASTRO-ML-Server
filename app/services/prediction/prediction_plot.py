import logging
import tempfile
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from plotly.subplots import make_subplots
from prophet import Prophet

from .utils import process_file


async def predict_plot(
    file: UploadFile = File(...),
    store_num: Optional[int] = Form(None),
    item_num: Optional[int] = Form(None),
    period_type: str = Form(...),
    num_periods: int = Form(...),
):
    try:
        df_data, period = await process_file(
            file, store_num, item_num, period_type, num_periods
        )

        model_prophet = Prophet()
        model_prophet.fit(df_data, seed=4)

        df_future = model_prophet.make_future_dataframe(
            periods=period, freq="D", include_history=False
        )

        forecast_prophet = model_prophet.predict(df_future)
        forecast_prophet.round()

        # Calculate the rolling average for the actual sales data
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

        # Create the plot
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=df_data["ds"],
                y=df_data["sales_trend"],
                name="Actual Sales",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_prophet["ds"],
                y=forecast_prophet["yhat_trend"],
                name="Forecast",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_prophet["ds"],
                y=forecast_prophet["yhat_upper_trend"],
                name="Upper Bound",
                fill="tonexty",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_prophet["ds"],
                y=forecast_prophet["yhat_lower_trend"],
                name="Lower Bound",
                fill="tonexty",
                showlegend=False,
            )
        )

        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmpfile:
            pio.write_image(fig, tmpfile.name)
            return FileResponse(tmpfile.name, media_type="image/svg+xml")

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
