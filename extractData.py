from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/ConstructionDB")

mydatabase = client.name_of_the_database
print(mydatabase.myTable)