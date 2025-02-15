from load_data_mongodb import app
from utils_and_constants import data_load, PATH, files
from fastapi import FastAPI
import pandas as pd


# FastAPI app instance
#app = FastAPI(title="Product classification",
#              description="API powered by FastAPI.",
#              version="1.0.1")


#@app.get("/upload_train_csv")
def get_data_train():
    """Please click on 'Try it out' and 'execute'
    to load a sample of the dataset
    Args:
    path and files to prepare text files

    Return:
    file in json format
    """
    file_name = pd.DataFrame.from_dict(data_load(PATH, files)[0][0])
    print("displaying only first 10 rows")
    file_name = file_name.head(10)
    file_name_Json = {
        "data": file_name.to_dict(orient="records"),  # Convert rows to [{}]
        "target": "target"
    }
    return file_name_Json
