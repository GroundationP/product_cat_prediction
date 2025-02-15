import io
from PIL import Image
from pymongo import MongoClient
from gridfs import GridFS

client = MongoClient(host="localhost", port=27017, username="li", password="pw")
db = client["rakuten_db"]
fs = GridFS(db)


file = fs.find_one({"filename": "image3.jpg"})
img_data = file.read()

# Display Image
image = Image.open(io.BytesIO(img_data))
image.show()





