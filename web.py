import os
import sys
import logging
from contextlib import asynccontextmanager
import joblib
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline.prediction_pipeline import ShipmentPricePredictor
from src.logger import logging

# AWS S3 Configuration (Best practice: Read from environment variables)
S3_BUCKET = os.getenv("BUCKET_NAME")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "best_model.pkl")
S3_PREPROCESSOR_KEY = os.getenv("S3_PREPROCESSOR_KEY", "preprocessing.pkl")

# Local Fallback Configuration
LOCAL_MODEL_PATH = "artifact/model_trainer/best_model.pkl"
LOCAL_PREPROCESSOR_PATH = "artifact/data_transform/transform_obj/preprocessing.pkl"

# Global application state dictionary to avoid unsafe global variables
app_state = {"pipeline": None}


def load_artifact_from_s3_or_local(s3_client, s3_key: str, local_path: str):
    """
    Attempts to stream/download an ML artifact from S3. 
    Falls back to the local file path if AWS fails.
    """
    try:
        logging.info(f"Attempting to fetch s3://{S3_BUCKET}/{s3_key}...")
        # Check if S3 client is initialized and bucket exists
        s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        # Load directly from memory stream to avoid unnecessary disk writes
        return joblib.load(s3_obj["Body"])
    except (NoCredentialsError, ClientError, Exception) as e:
        logging.warning(
            f"AWS S3 fetch failed for key '{s3_key}' due to: {e}. "
            f"Falling back to local path: '{local_path}'"
        )
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Critical Error: Fallback local file not found at {local_path}")
        return joblib.load(local_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan manager handling startup/shutdown routines safely.
    """
    # STARTUP ROUTINE
    try:
        # Initialize boto3 client. It automatically picks up AWS_ACCESS_KEY_ID, 
        # AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION from system env.
        s3_client = boto3.client("s3")
    except Exception:
        logging.warning("Failed to initialize boto3 client. Forcing local fallback mode.")
        s3_client = None

    try:
        trained_model = load_artifact_from_s3_or_local(s3_client, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        transform_object = load_artifact_from_s3_or_local(s3_client, S3_PREPROCESSOR_KEY, LOCAL_PREPROCESSOR_PATH)

        app_state["pipeline"] = ShipmentPricePredictor(
            transform_object=transform_object,
            trained_model=trained_model
        )
        logging.info("ShipmentPricePredictor successfully initialized.")
    except Exception as e:
        logging.critical(f"Application failed to initialize ML pipeline: {e}")
        # Crash the application immediately if models cannot be loaded anywhere
        sys.exit(1)
        
    yield
    # SHUTDOWN ROUTINE (Clean up resources here if needed)
    app_state.clear()


app = FastAPI(
    title="Shipment Price Prediction API",
    version="1.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Artist_Reputation: float = Form(...),
    Height: float = Form(...),
    Width: float = Form(...),
    Weight: float = Form(...),
    Material: str = Form(...),
    Price_Of_Sculpture: float = Form(...),
    Base_Shipping_Price: float = Form(...),
    International: str = Form(...),
    Express_Shipment: str = Form(...),
    Installation_Included: str = Form(...),
    Transport: str = Form(...),
    Fragile: str = Form(...),
    Customer_Information: str = Form(...),
    Remote_Location: str = Form(...),
    Scheduled_Date: str = Form(...),
    Delivery_Date: str = Form(...),
    Customer_Location: str = Form(...)
):
    try:
        if app_state["pipeline"] == None:
            raise HTTPException(status_code=503, detail="Model pipeline is uninitialized.")

        input_data = {
            "Artist Reputation": Artist_Reputation,
            "Height": Height,
            "Width": Width,
            "Weight": Weight,
            "Material": Material,
            "Price Of Sculpture": Price_Of_Sculpture,
            "Base Shipping Price": Base_Shipping_Price,
            "International": International,
            "Express Shipment": Express_Shipment,
            "Installation Included": Installation_Included,
            "Transport": Transport,
            "Fragile": Fragile,
            "Customer Information": Customer_Information,
            "Remote Location": Remote_Location,
            "Scheduled Date": Scheduled_Date,
            "Delivery Date": Delivery_Date,
            "Customer Location": Customer_Location
        }

        prediction = app_state["pipeline"].predict_single(input_data)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": f"${prediction:,.2f}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "healthy" if app_state["pipeline"] is not None else "unhealthy",
        "model_loaded": app_state["pipeline"] is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)