from pymongo import MongoClient 

client = MongoClient ('mongodb://localhost:27017/')

db = client ['Travel']

Pages = db ['Pages']
Information = db ['Information']
Reviews_Rating = db ['Reviews_Rating']
Url_Recommendations = db ['Url_Recommendations']
Agoda_Activities_Reviews = db ['Agoda_Activities_Reviews']