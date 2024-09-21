# Astro Demand Forecasting Server Using Meta's Prophet

We use docker to build and run a server that uses Meta's Prophet to forecast sales data and deploy it to Google Cloud Run.

## Prerequisites
- Ensure you have Docker installed and properly configured on your system.
---

## Steps

### 1. Build Local Docker Image

To build a Docker image :

```bash
docker build -t server_image . && docker image prune -f
```

### 2. Run the Docker Container Locally

To run the previously built `server_image` Docker container locally, use the following command:

```bash
docker rm -f ml_server && docker run --name ml_server -p 8080:8080 server_image
```

---

## API Endpoints

We Use FastAPI to create the API endpoints and Uvicorn to run the server.


## Additional Information

- [Docker](https://docs.docker.com/)
- [Prophet](https://facebook.github.io/prophet/docs/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Cloud Run](https://cloud.google.com/run)
