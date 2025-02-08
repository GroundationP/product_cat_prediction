import pytest
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/terence/A_NOTEBOOKS/Datasciencetest/PROJET_RAKUTEN/project/app"))
sys.path.append(src_path)  # Add 'src' to Python path
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import train_model
import utils_and_constants
from load_data import app

client = TestClient(app)

@pytest.fixture
def gen_hyper():
    hyper_p = {
      "max_depth": 6,
      "learning_rate": 0.1,
      "n_estimators": 100,
      "test_size": 0.2,
      "random_state": 42
    }
    return hyper_p
    

def test_train_valid_params(gen_hyper):
    """Test model training with valid hyperparameters."""
    response = client.post("/train?run_name_entry=pytest_model", json=gen_hyper)
    assert response.status_code == 200
    data_resp = response.json()
    assert "accuracy train" in data_resp
    assert "accuracy test" in data_resp
    assert isinstance(data_resp["accuracy train"], float)
    assert isinstance(data_resp["accuracy train"], float)



