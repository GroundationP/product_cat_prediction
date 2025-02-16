## Rakuten product prediction

This project aims to develop an API-driven solution for classifying electronic products based on textual descriptions. Using a dataset provided by the Rakuten Challenge https://challengedata.ens.fr/challenges/35 in CSV format the files and images were ingested into a MongoDB and the API use those files for building a Word2Vec model after normalization.

With FastAPI, a series of GET and POST endpoints were implemented to:

  -> Load the dataset.

  -> Train the XGBoost model with user-defined hyperparameters.
  
  -> Predict the category of a free-text input.
  
All experiments are logged using MLflow, enabling users to track hyperparameter evolution and model performance. For data versioning, DVC (Data Version Control) is used.


### Setup API
#### To create environment
```
python3 -m venv env_rakuten
cd env_rakuten/bin
source activate env_rakuten
pip install -r requirements.txt
```

#### To call process via bash
```
chmod +x run_server.sh (first time only)
./run_server.sh
```

#### To stop processes
```
cmd c
ps aux | grep "uvicorn\|mlflow"
pkill -f unicorn
kill 88895
```

#### to call webui
```
http://127.0.0.1:5000/
http://127.0.0.1:8000/docs
```

### Perform Pytest
```
python3 -m pytest tests/test_utils_and_constants.py
python3 -m pytest tests/test_train_model.py
```

### Versioning data with DVC
#### Create dvc env
```
python3 -m venv env_dvc
cd env_dvc/bin
source activate env_dvc
pip install dvc
```

#### to initialise (it's important to initialize it in the same folder from git cloned files)
```
git init
dvc init
```

#### To create a remote repository for files
```
mkdir ../dvc_remote
dvc remote add -d remote_storage ../dvc_remote
```

#### to track files (data file must be in a dvc file repository)
```
dvc add data/original
```
#### Adding files to staging area and push to git
```
git add data/.gitignore data/original.dvc
#git add app/data/.gitignore app/data/original.dvc (in this case git was initialized from a different folder look at __init__.py)
git commit -m "First data versioning with DVC (input files)"
dvc push
git push
```


#### To start docker MongoDB
```
docker-compose up -d
docker exec -it my_mongo bash 
mongosh -u li -p pw
db.fs.files.find()
```



