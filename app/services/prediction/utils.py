import io
import logging
from typing import Optional

import pandas as pd
from fastapi import HTTPException, UploadFile


async def process_file(
    file: UploadFile,
    store_num: Optional[int],
    item_num: Optional[int],
):
    logging.info("Store number: %s", store_num)
    logging.info("Item number: %s", item_num)

    contents = await file.read()
    logging.info("File content length: %d", len(contents))

    df_data = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    logging.info("Dataframe shape: %s", df_data.shape)
    logging.info("Dataframe columns: %s", df_data.columns.tolist())

    # Strip whitespace from column names
    df_data.columns = df_data.columns.str.strip()
    logging.info("Stripped Dataframe columns: %s", df_data.columns.tolist())

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

    return df_data
