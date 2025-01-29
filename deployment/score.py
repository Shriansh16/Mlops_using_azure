import json
import numpy
import joblib
import time
from azureml.core.model import Model

def init():
    global LGBM_MODEL
    try:
        # Load the model from file into a global object
        model_path = Model.get_model_path("insurance_model")
        print(f"Model path: {model_path}")  # Debugging
        LGBM_MODEL = joblib.load(model_path)
        print("Model loaded successfully.")  # Debugging
    except Exception as e:
        print(f"Error loading model: {str(e)}")  # Debugging
        raise e

def run(raw_data, request_headers):
    try:
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        print(f"Input data: {data}")  # Debugging
        result = LGBM_MODEL.predict(data)
        # Log the input and output data to appinsights:
        info = {
            "input": raw_data,
            "output": result.tolist()
        }
        print(json.dumps(info))  # Debugging
        print(('{{"RequestId":"{0}", '
               '"TraceParent":"{1}", '
               '"NumberOfPredictions":{2}}}'
               ).format(
                   request_headers.get("X-Ms-Request-Id", ""),
                   request_headers.get("Traceparent", ""),
                   len(result)
        ))
        return {"result": result.tolist()}
    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Debugging
        raise e