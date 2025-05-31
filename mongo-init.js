db = db.getSiblingDB('rakuten_db');
db.createCollection('dummy'); // creates the DB with a dummy collection
db.dummy.insert({ initialized: true });