"""
Here we generate embedding for the identified categories from GPT and M&S category
"""

import faiss
from sentence_transformers import SentenceTransformer

WW_CATEGORIES = [
    "dresses",
    "tops",
    "fascinators",
    "trainers",
    "trousers",
    "jumpsuits",
    "pinafores",
    "hair accessories",
    "bikini sets",
    "jumpers",
    "bras",
    "jeans",
    "shirts",
    "bags",
    "cardigans",
    "purses",
    "bikinis",
    "tankinis",
    "necklaces",
    "gloves",
    "sunglasses",
    "dungarees",
    "hats",
    "wellies",
    "umbrellas",
    "scarves",
    "bikini tops",
    "all in ones",
    "jersey shorts",
    "watches",
    "leggings",
    "belts",
    "jackets",
    "bracelets",
    "sport equipment",
    "skirts",
    "sweatshirts",
    "tunics",
    "rings",
    "skorts",
    "camisoles",
    "jeggings",
    "hoodies",
    "headbands",
    "culottes",
    "loafers",
    "vests",
    "shorts",
    "t-shirts",
    "crop tops",
    "coats",
    "shoes",
    "bikini bottoms",
    "jewellery sets",
    "polo shirts",
    "earrings",
    "accessories",
    "vest tops",
    "boots",
    "blazers",
    "slippers",
    "blouses",
    "joggers",
    "sandals",
    "gilets",
    "playsuits",
]


model = SentenceTransformer("sentence-transformers/average_word_embeddings_glove.6B.300d")
index = faiss.IndexHNSWFlat(300, 5000)

# ids = list(range(len(WW_CATEGORIES)))
# Sentences are encoded by calling model.encode()
embeddings = model.encode(WW_CATEGORIES)
index.add(embeddings)
print(index.ntotal)


def get_mns_category(category_from_gpt, model=model, index=index):
    query_embedding = model.encode([category_from_gpt])
    k = 5  # we want to see closest category
    D, I = index.search(query_embedding, k)  # actual search
    # top_index = I[0][0]
    mapped_category = WW_CATEGORIES[I[0][0]]

    print(f"{category_from_gpt} mapped to {mapped_category}")
    return mapped_category


if __name__ == "__main__":
    category = get_mns_category("white trainers")
    print(category)
