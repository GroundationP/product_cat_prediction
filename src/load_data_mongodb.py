from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from gridfs import GridFS
import io

# FastAPI app instance
app = FastAPI(title="Product classification",
              description="API powered by FastAPI.",
              version="1.0.1")


# MongoDB connection
client = MongoClient(host="localhost", port=27017, username="li", password="pw")
db = client["rakuten_db"]
collection = db["fs.files"]
fs = GridFS(db)

@app.get("/files")
def get_all_files():
    """Retrieve top 100 documents from mongodb. click on 'Try it out' and 'execute'"""
    files = collection.find()
    # Convert documents to list and ObjectId to string
    results = []
    for file in files[:100]:
        file["_id"] = str(file["_id"])
        results.append(file)
    return results
    
@app.get("/file/{key}")
def get_file_by_key(key: str):
    """Retrieve a document from Mongodb using the 'key'
    Args:
    value less than 85k

    Return:
    document description in json format
    """
    
    document = collection.find_one({"key": key})

    if not document:
        raise HTTPException(status_code=404, detail="File not found")

    # Convert ObjectId to string for JSON response
    document["_id"] = str(document["_id"])
    return document
    
@app.get("/image/{filename}")
async def get_image(filename: str):
    """Retrieve an image from Mongodb using the 'key'
    Args:
    value less than 85k

    Return:
    image in jpeg format
    """
    file = fs.find_one({"key": filename})
    if not file:
        raise HTTPException(status_code=404, detail="Image not found")    
    return StreamingResponse(io.BytesIO(file.read()), media_type="image/jpeg")
    
