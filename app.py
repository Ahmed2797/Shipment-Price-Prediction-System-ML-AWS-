import logging
import joblib
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline.prediction_pipeline import ShipmentPricePredictor

app = FastAPI(
    title="Shipment Price Prediction API",
    version="1.0"
)

# Mount static files directory for frontend assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize template engine
templates = Jinja2Templates(directory="templates")

# Global reference placeholder for the prediction pipeline
pipeline = None


@app.on_event("startup")
def load_model():
    """Initializes and caches the machine learning pipeline during application startup.

    Loads the pre-trained model and preprocessing transformer pipelines from 
    disk and instantiates the global `ShipmentPricePredictor`.
    """
    global pipeline
    try:
        trained_model = joblib.load("artifact/model_trainer/best_model.pkl")
        transform_object = joblib.load("artifact/data_transform/transform_obj/preprocessing.pkl")

        pipeline = ShipmentPricePredictor(
            transform_object=transform_object,
            trained_model=trained_model
        )
        logging.info("Model pipeline loaded successfully.")
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Renders the landing user interface form.

    Args:
        request (Request): The incoming FastAPI HTTP request context.

    Returns:
        HTMLResponse: Rendered template of the web form workspace.
    """
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
    """Handles web form submission, structures input features, and renders predictions.

    Args:
        request (Request): The request payload context for HTML rendering.
        Artist_Reputation (float): Normalized ranking of the artist.
        Height (float): Height dimension of the item.
        Width (float): Width dimension of the item.
        Weight (float): Direct mass weight of the shipment cargo.
        Material (str): Categorical composition material description (e.g., Wood, Marble).
        Price_Of_Sculpture (float): Declared monetary price or valuation of the asset.
        Base_Shipping_Price (float): Initial logistics baseline price quote.
        International (str): Destination cross-border status ('Yes'/'No').
        Express_Shipment (str): Expedited fulfillment status ('Yes'/'No').
        Installation_Included (str): Setup inclusion status ('Yes'/'No').
        Transport (str): Operational transportation mode choice.
        Fragile (str): Damage vulnerability tag ('Yes'/'No').
        Customer_Information (str): Classification designation for clients.
        Remote_Location (str): Remote geography accessibility indicator ('Yes'/'No').
        Scheduled_Date (str): Planned shipment start date string (YYYY-MM-DD).
        Delivery_Date (str): Planned shipment arrival date string (YYYY-MM-DD).
        Customer_Location (str): Destination location details.

    Returns:
        HTMLResponse: Re-rendered workspace template injected with computed cost predictions.
    """
    try:
        # Maps HTML form parameters directly to the training dataframe column schema keys
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

        # predict_single natively handles processing and target scale inversion transformations
        prediction = pipeline.predict_single(input_data)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": f"${prediction:,.2f}"  # Formats output gracefully as currency
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Performs a self-diagnostic check on structural API runtime status.

    Returns:
        dict: Diagnostic JSON object detailing readiness.
    """
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run using: uvicorn app:app --reload