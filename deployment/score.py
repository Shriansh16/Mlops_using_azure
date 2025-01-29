import json
import numpy as np
import joblib
import time
import os
import logging
from azureml.core.model import Model

def init():
    global LGBM_MODEL
    logging.info("Initializing model...")
    try:
        model_path = Model.get_model_path("insurance_model")
        logging.info(f"Model path: {model_path}")

        if not os.path.exists(model_path):
            logging.error("Model file not found at expected path.")
            raise FileNotFoundError(f"Model not found: {model_path}")

        LGBM_MODEL = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error during model initialization: {str(e)}")
        raise

def run(raw_data, request_headers):
    try:
        request_json = json.loads(raw_data)
        if "data" not in request_json:
            raise ValueError("Missing 'data' key in input JSON.")

        data = np.array(request_json["data"])
        result = LGBM_MODEL.predict(data)
        
        response = {"result": result.tolist()}
        logging.info(json.dumps(response))
        return response
    except Exception as e:
        logging.error(f"Error in run function: {str(e)}")
        return {"error": str(e)}