from src.entity.estimator import ShipmentPricePredictor
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

    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e


# input_data = {
#   "Artist_Reputation": 3.5,
#   "Height": 20,
#   "Width": 15,
#   "Weight": 50,
#   "Material": "Marble",
#   "Price_Of_Sculpture": 5000,
#   "Base_Shipping_Price": 200,
#   "International": "N0",
#   "Express_Shipment": "No",
#   "Installation_Included": "Yes",
#   "Transport": "Air",
#   "Fragile": "Yes",
#   "Customer_Information": "Premium",
#   "Remote_Location": "No"
# }
input_data = {
    "Customer Id": "fffe3900350033003300",
    "Artist Name": "Billy Jenkins",
    "Artist Reputation": 0.26,
    "Height": 17.0,
    "Width": 6.0,
    "Weight": 4128.0,
    "Material": "Brass",
    "Price Of Sculpture": 13.91,
    "Base Shipping Price": 16.27,
    "International": "Yes",
    "Express Shipment": "Yes",
    "Installation Included": "No",
    "Transport": "Airways",
    "Fragile": "No",
    "Customer Information": "Working Class",
    "Remote Location": "No",
    "Scheduled Date": "06/07/15",
    "Delivery Date": "06/03/15",
    "Customer Location": "New Michelle, OH 50777",
    # "Cost": -283.29
}

prediction = pred_model(input_data=input_data)
print(prediction)