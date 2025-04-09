from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from typing import List, Annotated
import os
import json
import requests
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from logger import Logger

MODEL_VERSION = os.getenv("MODEL_VERSION", "2025_04_01")
checkpoint_path = os.path.join('.','model_' + MODEL_VERSION + '_assets','checkpoints')
checkpoint_name = 'best_model.keras'
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/")
LOG_FILENAME = os.getenv("LOG_FILENAME", "log.txt")

test_infer_body = {
    'sepal_length': 6.3,
    'sepal_width': 2.5,
    'petal_length': 5.0,
    'petal_width': 1.9
}

test_infer_response = {
    "result": {
        "setosa": 0.027633268386125565,
        "versicolor": 0.31605255603790283,
        "virginica": 0.6563141345977783
    }
}

prob_tol = 0.1

# TODO: Add script to generate expected infer response for new model
# the expected infer response varies for different models
# due to the parallel nature of tensorflow training, even if we fix
# all the initial conditions including initial weights, the resulting
# model would still be slightly different. Thus, we need to generate
# new test case for each model.

class InputFeature(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def feature_to_array(input_feature):
    array = np.array(
        [
            input_feature.sepal_length,
            input_feature.sepal_width,
            input_feature.petal_length,
            input_feature.petal_width
        ]
    )
    return array

def output_array_to_dict(output_array):
    output_dict = {
        'setosa': float(output_array[0]),
        'versicolor': float(output_array[1]),
        'virginica': float(output_array[2])
    }
    return output_dict

nn = tf.keras.models.load_model(os.path.join(checkpoint_path, checkpoint_name))
logger = Logger(log_filename=LOG_FILENAME)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains like ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: add model version
# TODO: add rate limiter

@app.get('/status')
def check_status():
    try:
        response = requests.post(BASE_URL + "infer", json=test_infer_body)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Model endpoint unreachable")

        output = response.json()
        output_result = json.loads(output['result'])
        expected_result = test_infer_response['result']
        for key in output_result:
            if key not in expected_result:
                raise ValueError("Model endpoint returned result with unexpected key: " + key)
        for key in expected_result:
            if key not in output_result:
                raise ValueError("Model endpoint returned result missing key: " + key)
            if abs(output_result[key] - expected_result[key]) > prob_tol:
                raise ValueError("Probabilities don't match for key " + key + ". "
                                 "Expected " + str(expected_result[key]) + " found " + str(output_result[key])
                                )
        return {"message": "Inference endpoint status OK"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("./static/index.html", "r") as f:
        return f.read()

# dynamically serve config.js, now the base_url has single
# source of the truth, which is environmental variable
@app.get("/config.js")
def serve_config():
    js = f"const BASE_URL = '{BASE_URL}';"
    return Response(content=js, media_type="application/javascript")

@app.post("/infer")
def infer(input_feature: Annotated[InputFeature, Body()]):
    x = feature_to_array(input_feature).reshape(1,-1)
    logger.set_system_avg_cpu_percent_start_point()
    logger.set_proc_avg_cpu_percent_start_point()
    # the cpu percent measurement may be inaccurate if the
    # prediction process is too fast. In this case, for
    # debugging purpose, we can repeat the process many times
    # for i in range(100):
    #     probs = safe_predict(x)[0].numpy()
    probs = safe_predict(x)[0].numpy()
    logger.log_proc_avg_cpu_percent()
    logger.log_system_avg_cpu_percent()
    logger.log_proc_memory()
    logger.log_system_memory()
    output_dict = output_array_to_dict(probs)
    return {'result':json.dumps(output_dict)}

# tf.function decorator avoids re-creating new computational 
# graphs every time a function runs
@tf.function
def safe_predict(x):
    return nn(x)