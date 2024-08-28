# Astro Demand Forecasting Server Using Meta's Prophet

This documentation outlines the process for building, tagging, and deploying Docker images for the Astro-ML project, using Docker and Google Cloud.

## Prerequisites

- Ensure you have Docker installed and properly configured on your system.
- Google Cloud SDK (`gcloud`) is installed and configured with your project.
- Make sure you are authenticated and have access to the Google Container Registry.

---

## Steps

### 1. Build Local Docker Image

To build a Docker image specifically for an ARM architecture:

```bash
docker build -t server_image_arm . && docker image prune -f
```

### 2. Run the Docker Container Locally

To run the previously built `server_image_arm` Docker container locally, use the following command:

```bash
docker rm -f ml_server && docker run --name ml_server -p 8080:8080 server_image_arm
```

This will start the container on port `8080` and remove any existing containers with the same name (`ml_server`).

### 3. Authenticate Google Cloud Docker Registry

Ensure you are authenticated with Google Cloud to push images to the Google Container Registry:

```bash
gcloud auth login
gcloud auth configure-docker asia-south1-docker.pkg.dev
```

### 4. Tag and Push the Docker Image

Once your image is built, you can tag it for pushing to the Google Cloud Registry:

```bash
docker tag server_image $PUSH_REPO
```

### 5. Build and Push Multi-Platform Docker Image

To build and push a Docker image targeting the `linux/amd64` platform and store it in the Google Container Registry, use the following command:

```bash
docker buildx build --platform linux/amd64 -t $PUSH_REPO --push .
```

---

## Notes

- Ensure your `gcloud` account has the necessary permissions to push to the Google Cloud Registry.
- Adjust the `asia-south1` region to match the region you're using for your Google Cloud project.

---

## API Endpoints

We Use FastAPI to create the API endpoints and Uvicorn to run the server.

### 1. Predict Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Upload a CSV file containing sales data to get a sales forecast.

#### Expected CSV Format

- `date`: Date of the sales record.
- `store`: Store number.
- `item`: Item number.
- `sales`: Sales amount.

#### Example JSON Request using Python

Here is an example of how to make a request to the `/predict` endpoint using Python and the `requests` library:

```python
import requests

url = "http://localhost:8080/predict"
files = {'file': open('path/to/your/file.csv', 'rb')}
data = {
    'store_num': 1,
    'item_num': 1,
    'period_type': 'M',
    'num_periods': 3
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Example JSON Response

The response will be a JSON array of predictions. Here is an example response:

```json
[
    {
        "ds": "2018-01-01T00:00:00.000",
        "trend": 22.7140167564,
        "yhat_lower": 7.4571324502,
        "yhat_upper": 18.4179830033,
        "trend_lower": 22.7140167564,
        "trend_upper": 22.7140167564,
        "additive_terms": -9.7333522074,
        "additive_terms_lower": -9.7333522074,
        "additive_terms_upper": -9.7333522074,
        "weekly": -4.4043610074,
        "weekly_lower": -4.4043610074,
        "weekly_upper": -4.4043610074,
        "yearly": -5.3289912,
        "yearly_lower": -5.3289912,
        "yearly_upper": -5.3289912,
        "multiplicative_terms": 0.0,
        "multiplicative_terms_lower": 0.0,
        "multiplicative_terms_upper": 0.0,
        "yhat": 12.980664549
    },
    {
        "ds": "2018-01-02T00:00:00.000",
        "trend": 22.7159111136,
        "yhat_lower": 9.5803628246,
        "yhat_upper": 20.9826524677,
        "trend_lower": 22.7159111136,
        "trend_upper": 22.7159111136,
        "additive_terms": -7.0786119915,
        "additive_terms_lower": -7.0786119915,
        "additive_terms_upper": -7.0786119915,
        "weekly": -1.7875504981,
        "weekly_lower": -1.7875504981,
        "weekly_upper": -1.7875504981,
        "yearly": -5.2910614933,
        "yearly_lower": -5.2910614933,
        "yearly_upper": -5.2910614933,
        "multiplicative_terms": 0.0,
        "multiplicative_terms_lower": 0.0,
        "multiplicative_terms_upper": 0.0,
        "yhat": 15.6372991221
    }
]
```

## Additional Information

- [Docker](https://docs.docker.com/)
- [Prophet](https://facebook.github.io/prophet/docs/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
