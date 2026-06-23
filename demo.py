from src.pipeline.prediction_pipeline import ShipmentPricePredictor
import logging
import joblib


def pred_model(input_data: dict):
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
        prediction = pipeline.predict_single(input_data)

        return prediction

        logging.info("Model pipeline loaded successfully.")
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e


input_data = {
  "Artist_Reputation": 3.5,
  "Height": 20,
  "Width": 15,
  "Weight": 50,
  "Material": "Marble",
  "Price_Of_Sculpture": 5000,
  "Base_Shipping_Price": 200,
  "International": "N0",
  "Express_Shipment": "No",
  "Installation_Included": "Yes",
  "Transport": "Air",
  "Fragile": "Yes",
  "Customer_Information": "Premium",
  "Remote_Location": "No"
}

prediction = pred_model(input_data=input_data)
print(prediction)