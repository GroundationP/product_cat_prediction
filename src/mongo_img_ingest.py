from pymongo import MongoClient
from gridfs import GridFS
import pickle 

client = MongoClient(host="localhost", port=27017, username="li", password="pw")
db = client["rakuten_db"]
fs = GridFS(db)

# importing dictionnary
with open('/Users/terence/A_NOTEBOOKS/Datasciencetest/PROJET_RAKUTEN/project/data/original/rak_txt_img_dict.pkl', 'rb') as f:
    rak_txt_img_dict = pickle.load(f)

# ingesting data into Mongodb 
for k, v in rak_txt_img_dict.items():
    with open(f"/Users/terence/A_NOTEBOOKS/Datasciencetest/PROJET_RAKUTEN/images/image_train/{v[0]}", "rb") as r:
        file_id = fs.put(r, key=f"{k}", imageid=f"{v[0]}", prdtypecode=f"{v[1]}", designation=f"{v[2]}", description=f"{v[3]}")        
    #print(f"File stored with ID: {file_id}")




