from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

from predict_examples import EXAMPLE_1, EXAMPLE_2, EXAMPLE_3
from help_funcs import search_rows, col_to_feat

app = FastAPI()

# Serve /interactives/* (scatter plots .html)
app.mount("/interactives", StaticFiles(directory="interactives"), name="interactives")

@app.get("/", response_class=HTMLResponse)
def home():
    # Serve one of your pages as the default landing
    with open("interactives/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get(
    "/examples/1",
    response_model=dict[str, Any],
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": EXAMPLE_1
                }
            }
        }
    },
)
def example1():
    return EXAMPLE_1

@app.get(
    "/examples/2",
    response_model=dict[str, Any],
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": EXAMPLE_2
                }
            }
        }
    },
)
def example2():
    return EXAMPLE_2

@app.get(
    "/examples/3",
    response_model=dict[str, Any],
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": EXAMPLE_3
                }
            }
        }
    },
)
def example3():
    return EXAMPLE_3
    
    
@app.post("/search")
def search(
    s: str =Body(
        default="skyrim",
        description="Fuzzy search is supported.",
    )
):
    matches = search_rows(df_results, "name", s,fuzzy_threshold=75)
    pd.options.display.float_format = "{:.1f}".format
    return matches.to_dict('records')
    
@app.post("/predict")
def predict(
    x: dict[str, Any] = Body(
        default=EXAMPLE_1,  # <-- this is the key
        description="You can find more examples in /examples/2 and /examples/3",
    )
):
    # Build X exactly how your model expects it
    df = pd.DataFrame([x])
    df = df.reindex(columns=EXPECTED_COLS)

    df_X_z = col_to_feat(df)
    all_X = df_X_z.to_numpy(dtype="float32")
    y_pred = SS_y.inverse_transform(model.predict(all_X).reshape(-1, 1)).squeeze(-1)
    
    p_pred = confmodel.predict_proba(df_X_z)[:, 1]
    if p_pred>0.7:
        conf_tier = 4
        conf_label = "very high"
    elif p_pred>0.4:
        conf_tier = 3
        conf_label = "high"
    elif p_pred>0.1:
        conf_tier = 2
        conf_label = "medium"
    else:
        conf_tier = 1
        conf_label = "low"

    return {"pred_score":float(y_pred[0]),"conf_tier":conf_tier,"conf_label":conf_label}

# /docs will exist automatically
