from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/")
def root():
    """
    Handles HTTP GET requests to the root endpoint ('/').

    Returns:
        dict: A JSON response containing a welcome message.
    """
    return {"message": "Welcome to the FastAPI!"}


@router.get("/info")
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
