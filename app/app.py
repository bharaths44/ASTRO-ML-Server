
import logging
from fastapi import FastAPI
from app.routes import main_routes, predict_routes
app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app.include_router(main_routes.router)
app.include_router(predict_routes.router)
