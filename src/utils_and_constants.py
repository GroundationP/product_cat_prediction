import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
from PIL import Image
from IPython.display import display
import re
import joblib
from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from base64 import b64decode
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import io
import csv
import random
from io import StringIO
import mlflow
import mlflow.xgboost
import pickle
import joblib
import json


# Define the data schema
class TrainData(BaseModel):
    data: list[dict]  # List of dictionaries for tabular data
    target: str       # Target column name

# Define request model for training parameters
class TrainRequest(BaseModel):
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 100
    test_size: float = 0.2
    random_state: int = 42

def clean_text(txt_file):
    """ This function is designed to clean text by eliminate strange characters and avoid biais while training a model.
    The fields concerned are 'designation' & 'description' fields.

    Args:
    txt_file (DataFrame) having the columns 'designation' & 'description'

    Return:
    txt_file (DataFrame) having the cleaned columns 'designation_clean' & 'description_clean'

    """
    txt_file = txt_file.fillna("NULL")
    for d in txt_file.columns:
        #print(d)
        txt_file["{}_clean".format(d)] = txt_file[d].str.lower()
        #### to eliminate < bla bla bla >
        pattern1 = r"<[^>]*>"
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].apply(lambda x: re.sub(pattern1, "", x))
        #### to eliminate strange characters
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\(','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\)','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\'','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\]','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\[','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('n°','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('dun','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('lot','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('of','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('cm','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('mm','', regex=True)
        # To avoid single letter in the text description
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].apply(lambda x: re.sub(r'\b[a-zA-Z]\b', '', x)).apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        # To avoid single numbers in the text description
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].apply(lambda x: re.sub(r'\b\d\b', '', x)).apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('@','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('#','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('&','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\'','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('!','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('-','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('_','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('$','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\*','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('^','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('%','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('`','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('=','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\+','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(':','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('/','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(';','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\.','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(',','', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace('\?','', regex=True)
        #### to eliminate accents
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'š','s', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ð','dj', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ž','z', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'à','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'á','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'â','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ã','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ä','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'å','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'æ','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ç','c', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'è','e', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'é','e', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ê','e', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ë','e', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ì','i', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'í','i', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'î','i', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ï','i', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'i','i', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ñ','n', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ò','o', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ó','o', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ô','o', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'õ','o', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ö','o', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ø','o', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ù','u', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ú','u', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'û','u', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ü','u', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'þ','b', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ß','ss', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ğ','g', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ð','o', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ñ','n', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ş','s', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ý','y', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ÿ','y', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ƒ','f', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ł','l', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ź','z', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ś','s', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ń','n', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ą','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ż','z', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ę','e', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ő','o', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ī','i', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ā','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ē','e', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ķ','k', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ė','e', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ū','u', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ș','s', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ț','t', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ă','a', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'ć','c', regex=True)
        txt_file["{}_clean".format(d)] = txt_file["{}_clean".format(d)].replace(u'č','c', regex=True)
    return txt_file
    

# fonction to vectorize words
def make_sentence_vector(sentence, word_dict):
    idxs = [word_dict[w] for w in sentence if w in word_dict]
    return idxs


# fonction to complete with 0 the max length
def def_pad(sentence, pad_length, p):
    miss_zero = pad_length - len(sentence)
    return sentence + [p] * miss_zero
    

# Function to limit string length to 300 words
def limit_words(text, max_words=200):
    words = text.split()  # Split into words
    return ' '.join(words[:max_words])  # Keep only the first 300 words


def data_load(PATH, files):
    global corpus_desig_json
    # Loading X_Train, y_Train and X_Test in CSV format
    X_Train = pd.read_csv(PATH+'{}'.format(files[0]), encoding='utf-8').rename(columns={'Unnamed: 0': 'uid'})#.head(100)
    Y_Train = pd.read_csv(PATH+'{}'.format(files[1]), encoding='utf-8').rename(columns={'Unnamed: 0': 'uid'})#.head(100)
    X_test_update = pd.read_csv(PATH+'{}'.format(files[2]), encoding='utf-8').rename(columns={'Unnamed: 0': 'uid'})#.head(100)
    # merging X_Train and Y_Train dataset
    X_Train = pd.merge(X_Train, Y_Train, left_on='uid', right_on='uid', how='left')
    # Limiting to the first 200 words found in 'designation'
    X_Train['designation'] = X_Train['designation'].apply(limit_words)
    # Generating corpus file
    corpus_desig = pd.concat([X_Train[['designation']], X_test_update[['designation']]])
    # Converting files to json format
    X_Train_json = X_Train.to_json(orient="records")
    X_test_update_json = X_test_update.to_json(orient="records")
    corpus_desig_json = corpus_desig.to_json(orient="records")
    # Convert and write JSON object to file (double addresses because of api and pytest versions)
    try:
        with open("app/models/corpus_desig.json", "w") as outfile: 
            json.dump(corpus_desig_json, outfile)
    except:
        with open("models/corpus_desig.json", "w") as outfile: 
            json.dump(corpus_desig_json, outfile)        
    # call function data_eng to clean text
    X_Train = data_eng(X_Train_json, corpus_desig_json)
    X_test_update = data_eng(X_test_update_json, corpus_desig_json)
    return [X_Train, X_test_update, corpus_desig_json]


def data_eng(file_name_json, corpus_desig_json):
    global word_dict
    global le
    # from json to dataframe
    file_name = pd.read_json(StringIO(file_name_json), orient='records')
    corpus_desig = pd.read_json(StringIO(corpus_desig_json), orient='records')
    # cleaning texts
    # file_name_clean = clean_text(file_name[['designation', 'description']]).drop(['designation', 'description'], axis=1)
    file_name_clean = clean_text(file_name[['designation']]).drop(['designation'], axis=1)
    file_name = pd.concat([file_name, file_name_clean], axis=1)
    corpus_desig = clean_text(corpus_desig[['designation']]).drop(['designation'], axis=1)
    # tokenizing words
    file_name['designation_clean_token'] = file_name['designation_clean'].map(word_tokenize)
    corpus_desig['designation_clean_token'] = corpus_desig['designation_clean'].map(word_tokenize)
    # Vectorizing words
    unique_tokens = []
    for l1 in corpus_desig['designation_clean_token'].tolist():
        for l2 in l1:
            unique_tokens.append(l2)
    corpus = list(set(unique_tokens))
    corpus_length = 300 # 300 # max size corpus or len(corpus)
    inverse_word_dict = {}
    for i, word in enumerate(corpus):
        i+=len(word_dict) + 1
        word_dict[word] = i
        inverse_word_dict[i] = word
    file_name['designation_vector'] = file_name['designation_clean_token'].apply(lambda x: make_sentence_vector(x, word_dict))
    corpus_desig['len_designation_clean'] = corpus_desig['designation_clean'].map(len)
    # Padding with 0 missing values
    p = 0
    pad_length = corpus_desig['len_designation_clean'].max()
    file_name['designation_vector'] = file_name['designation_vector'].apply(lambda x: def_pad(x, pad_length, p))
    # dealing with labels
    try:
        y_file_name = le.fit_transform(file_name['prdtypecode'])
        try:
            filename = 'app/models/label_encoding.pkl'
            joblib.dump(le, filename)
        except:
            filename = 'models/label_encoding.pkl'
            joblib.dump(le, filename)
    except:
        pass
    # exploding (vector to columns)
    file_name_List = []
    for i in file_name['designation_vector']:
        file_name_List.append(i)
    file_name_df = pd.DataFrame(file_name_List)
    try:
        # train/test flag
        file_name_df['target'] = y_file_name
        # Convert the DataFrame to the desired JSON format to API
        file_name_Json_final = {
            "data": file_name_df.to_dict(orient="records"),  # Convert rows to [{}]
            "target": "target"
        }
    except:
        # Convert the DataFrame to the desired JSON format to API
        file_name_Json_final = {
            "data": file_name_df.to_dict(orient="records")
        }
    return [file_name_Json_final, corpus_desig]


# making global some variables
xgb = None
corpus_desig_json = pd.DataFrame()
word_dict = {}
le = LabelEncoder()

# Load the CSV file
PATH = "address.../project/data/original/"
files = ['X_train_update_copil.csv', 'Y_train_update_copil.csv', 'X_test_update.csv']  # large file
files = ['X_train_update.csv', 'Y_train_CVw08PX.csv', 'X_test_update.csv']
