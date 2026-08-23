from src.entity.estimator import ShipmentPricePredictor
import logging
import joblib


def pred_model(input_data: dict):
    """Initializes and caches the machine learning pipeline during application startup.

    Loads the pre-trained model and preprocessing transformer pipelines from 
    disk and instantiates the global `ShipmentPricePredictor`.
    """
    try:
        trained_model = joblib.load("aws_model_init/best_model.pkl")
        transform_object = joblib.load("artifact/data_transform/transform_obj/preprocessing.pkl")
        power_transformer = joblib.load("artifact/data_transform/transform_obj/power_transformer.pkl")

        pipeline = ShipmentPricePredictor(
            transform_object=transform_object,
            power_transformer=power_transformer,
            trained_model=trained_model
        )
        prediction = pipeline.predict_single(input_data)

        return prediction

    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e


shipment_data = {
    "Customer Id": "fffe3800330031003900",
    "Artist Name": "Jean Bryant",
    "Artist Reputation": 0.28,
    "Height": 3.0,
    "Width": 3.0,
    "Weight": 61.0,
    "Material": "Brass",
    "Price Of Sculpture": 6.83,
    "Base Shipping Price": 15.0,
    "International": "No",
    "Express Shipment": "No",
    "Installation Included": "No",
    "Transport": "Roadways",
    "Fragile": "No",
    "Customer Information": "Working Class",
    "Remote Location": "No",
    "Scheduled Date": "03/06/17",
    "Delivery Date": "03/05/17",
    "Customer Location": "New Michaelport, WY 12072"
    ## "Cost": -159.96
}
input_data = {
    "Customer Id": "fffe3900350033003300",
    "Artist Name": "Billy Jenkins",
    "Artist Reputation": 0.28,
    "Height": 3.0,
    "Width": 3.0,
    "Weight": 61.0,
    "Material": "Brass",
    "Price Of Sculpture": 6.91,
    "Base Shipping Price": 15.00,
    "International": "No",
    "Express Shipment": "No",
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

shipment_data2 = {
    "Customer Id": "fffe3800360033003700",
    "Artist Name": "David Hawes",
    "Artist Reputation": 0.64,
    "Height": 17.0,
    "Width": 9.0,
    "Weight": 7264.0,
    "Material": "Brass",
    "Price Of Sculpture": 8.26,
    "Base Shipping Price": 90.67,
    "International": "No",
    "Express Shipment": "Yes",
    "Installation Included": "No",
    "Transport": "Roadways",
    "Fragile": "No",
    "Customer Information": "Working Class",
    "Remote Location": "No",
    "Scheduled Date": "06/05/16",
    "Delivery Date": "06/02/16",
    "Customer Location": "South Matthew, WV 76033"
##   "Cost": -1536.66
}

prediction = pred_model(input_data=shipment_data2)
print(prediction)