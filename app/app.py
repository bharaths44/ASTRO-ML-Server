# app.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
from forecast.forecast import graph_data
from preprocessing.create_predict_df import create_predict_df
from preprocessing.date_feature import create_date_features as date_feature
from preprocessing.lag_feature import lag_features
from preprocessing.ewm_feature import ewm_features
from preprocessing.roll_mean_feature import roll_mean_features
from feature_extraction.important_feature import features

app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

loaded_model = lgb.Booster(model_file="final_model.txt")


def preprocess(data):
    df = pd.DataFrame(data)
    df = date_feature(df)
    df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
    df = roll_mean_features(df, [365, 546, 730])
    df = ewm_features(
        df, [0.95, 0.9, 0.8, 0.7, 0.5], [91, 98, 105, 112, 119, 126, 182, 364, 546, 728]
    )
    return df


@app.get("/")
async def root():
    return "Welcome to the FastAPI!"


@app.get("/info")
async def info():
    return JSONResponse(
        {
            "name": "FastAPI",
            "version": "1.0",
            "description": "This is a FastAPI for predictions.",
        }
    )


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    store_num = data.get("store_num")
    item_num = data.get("item_num")
    period_type = data.get("period_type")
    num_periods = data.get("num_periods")

    data = pd.DataFrame(data.get("data"))
    predict_df = create_predict_df(data, period_type, num_periods)
    merge_data = pd.concat([data, predict_df], sort=False)
    merge_data = pd.DataFrame(merge_data)
    df = preprocess(merge_data)
    pred = features(df, loaded_model)
    val = loaded_model.predict(pred, predict_disable_shape_check=True, verbose=2)
    val = np.expm1(val)
    response = graph_data(predict_df, val, store_num, item_num, data=data)

    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
