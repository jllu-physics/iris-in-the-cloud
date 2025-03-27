from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Annotated
import json
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


checkpoint_name = 'best_model.keras'

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

nn = tf.keras.models.load_model("./checkpoints/" + checkpoint_name)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains like ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/infer")
def infer(input_feature: Annotated[InputFeature, Body()]):
    x = feature_to_array(input_feature).reshape(1,-1)
    probs = nn.predict(x)[0]
    output_dict = output_array_to_dict(probs)
    return {'result':json.dumps(output_dict)}