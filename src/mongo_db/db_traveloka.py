from pymongo import MongoClient

client = MongoClient ('mongodb://localhost:27017/')

db = client ['Traveloka']

Province_URL_Activities = db ['Pronvince_URL_Activities']
Activities_Reviews = db['Activities_Reviews']