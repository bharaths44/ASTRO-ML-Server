from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.services.prediction.prediction_data import predict_data


router = APIRouter()


@router.post("/predict/data")
async def predict_data_route(
    file: UploadFile = File(...),
    store_num: Optional[int] = Form(None),
    item_num: Optional[int] = Form(None),
):
    """
    Endpoint to predict data.

    Args:
        file (UploadFile): The CSV file containing the data.
        store_num (Optional[int]): The store number.
        item_num (Optional[int]): The item number.
        period_type (str): The type of period (e.g., 'M' for month).
        num_periods (int): The number of periods to predict.

    Returns:
        JSONResponse: The predicted data in JSON format.
    """
    return await predict_data(file, store_num, item_num)
