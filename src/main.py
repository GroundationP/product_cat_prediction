from load_data import app
from train_model import train_model
from utils_and_constants import data_eng
from fastapi import HTTPException
import pandas as pd
import joblib
import json


# Endpoint to predict with the trained model
@app.post("/predict")
def predict(data: list[dict]):
    """Before predict, please make sure the model is trained.
    The input format must follow the schema [{"designation":"text"}] 
    Args:
    data as list of dictionnaire. ex:
    [{"designation":"Olivia: Personalisiertes Notizbuch, 150 Seiten,
    Punktraster, Ca Din A5, Rosen-Design"}]

    Return:
    predicted class
    """

    filename = 'models/best_xgb.pkl'
    xgb = joblib.load(filename)
    if not xgb:
        raise HTTPException(status_code=400, detail="Model is not trained yet")
    # Convert input data to DataFrame
    with open('models/corpus_desig.json', 'r') as file:
        corpus_desig_json = json.load(file)
    input_data = data_eng(pd.DataFrame.from_dict(
                         data).to_json(orient="records"),
                         corpus_desig_json)[0]
    input_data = pd.DataFrame.from_dict(input_data['data'])
    # Ensure the model can predict on the given data
    try:
        filename = 'app/models/label_encoding.pkl'
        le = joblib.load(filename)
    except: 
        filename = 'models/label_encoding.pkl'
        le = joblib.load(filename)
    try:
        predictions = le.inverse_transform(xgb.predict(input_data))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predictions": predictions.tolist()}
