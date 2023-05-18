from pymongo import MongoClient
import os
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

MONGO_CONN_STR = os.environ["MONGO_STR"]

def load_data(chroma_client):
    print('attempting to load data')
    client = MongoClient(MONGO_CONN_STR)
    article_db = client["article-data"]
    article_collection = article_db["article-backup"]

    query = {"isActive": True, "inStock": True, "masterCategoryBreadcrumb": {"$regex": "^Women"},
         "colourName":{"$ne": None}, "category":{"$ne": None}, "gender":{"$ne": None}, "title":{"$ne": None}, "carouselImageUrl":{"$ne": None}}

    cursor = article_collection.find(query)

    documents = []
    metadatas = []
    article_ids = []

    for article in tqdm(cursor):
        article_ids.append(article.get("_id"))
        documents.append(article.get("description"))
        metadatas.append(
            {
                "article_id": article.get("_id"),
                "colour": article.get("colourName"),
                "category": article.get("category"),
                "gender": article.get("gender"),
                "title": article.get("title"),
                "imageUrl": article.get("carouselImageUrl"),
            }
        )

    print("Adding WW articles into collection")

    client = chroma_client
    # WARNING
    # client.delete_collection(name="test")

    collection = client.get_or_create_collection(name="test")

    # Note this is using BERT SentenceTransformerEmbeddingFunction
    collection.add(documents=documents, metadatas=metadatas, ids=article_ids)
    print('finished loading data')

if __name__ == "__main__":
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./.chroma/persist"))
    load_data(chroma_client=client)