import os
import sys
from contextlib import asynccontextmanager
import joblib
import boto3
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.entity.estimator import ShipmentPricePredictor
from src.logger import logging


# AWS S3 Configuration
S3_BUCKET = os.getenv("BUCKET_NAME")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "Best_Model/best_model.pkl")
S3_PREPROCESSOR_KEY = os.getenv("S3_PREPROCESSOR_KEY", "PREPROCESSING_OBJ/preprocessing.pkl")

# 1. NEW: Define our dedicated cloud download directory using Path
AWS_DOWNLOAD_DIR = Path("aws_model_init")

# Local Fallback Configuration (Kept as pure strings for maximum wrapper compatibility)
LOCAL_MODEL_PATH = "artifact/model_trainer/best_model.pkl"
LOCAL_PREPROCESSOR_PATH = "artifact/data_transform/transform_obj/preprocessing.pkl"

# 2. Update local paths to point inside our new directory
LOCAL_MODEL_PATH = AWS_DOWNLOAD_DIR / "best_model.pkl"
LOCAL_PREPROCESSOR_PATH = AWS_DOWNLOAD_DIR / "preprocessing.pkl"

# Global application state dictionary to avoid thread-unsafe variables
app_state = {"pipeline": None}


def load_artifact_from_s3_or_local(s3_client, s3_key: str, target_local_path: Path):
    """
    Downloads an ML artifact from AWS S3 to a local folder, then loads it.
    Falls back to existing local files if AWS is unreachable.
    """
    try:
        # Create the 'aws_model_init' directory automatically if it doesn't exist yet
        if not target_local_path.parent.exists():
            logging.info(f"Creating new directory: {target_local_path.parent}")
            target_local_path.parent.mkdir(parents=True, exist_ok=True)

        if not S3_BUCKET or not s3_client:
            raise ValueError("S3 Bucket name or AWS Client is not configured.")

        logging.info(f"Downloading s3://{S3_BUCKET}/{s3_key} to {target_local_path}...")
        
        # Download the file directly from S3 to your new local path
        s3_client.download_file(S3_BUCKET, s3_key, str(target_local_path))
        logging.info(f"Download complete. Loading asset into memory...")
        
        return joblib.load(str(target_local_path))
        
    except (NoCredentialsError, ClientError, Exception) as e:
        logging.warning(
            f"AWS S3 download failed for key '{s3_key}' due to: {e}. "
            f"Checking for existing files at: '{target_local_path}'"
        )
        
        # Fallback check: If the download failed, see if the file is already there from an older run
        if not target_local_path.exists():
            raise FileNotFoundError(
                f"Critical Error: Asset missing in cloud, and no fallback found at {target_local_path}"
            )
            
        return joblib.load(str(target_local_path))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan manager handling model caching during container bootstrap.
    """
    # --- STARTUP TRIGGER ---
    try:
        # Boto3 implicitly gathers AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY from env
        s3_client = boto3.client("s3")
    except Exception as e:
        logging.warning(f"Unable to initialize boto3 client: {e}. Defaulting straight to local storage fallback.")
        s3_client = None

    try:
        trained_model = load_artifact_from_s3_or_local(s3_client, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        transform_object = load_artifact_from_s3_or_local(s3_client, S3_PREPROCESSOR_KEY, LOCAL_PREPROCESSOR_PATH)

        app_state["pipeline"] = ShipmentPricePredictor(
            transform_object=transform_object,
            trained_model=trained_model
        )
        logging.info("ShipmentPricePredictor operational state reached successfully.")
    except Exception as e:
        logging.critical(f"Forced API shutdown: Failure loading underlying model stack. Details: {e}")
        sys.exit(1)
        
    yield
    # --- SHUTDOWN TRIGGER ---
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
        if app_state["pipeline"] is None:
            raise HTTPException(status_code=503, detail="Prediction service context uninitialized.")

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
        logging.error(f"Prediction route processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    is_healthy = app_state["pipeline"] is not None
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": is_healthy
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# Run using: uvicorn app:app --reload