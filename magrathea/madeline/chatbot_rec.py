import chromadb
from chromadb.config import Settings
from chromadb.errors import NoDatapointsException

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from magrathea.madeline.query import product_desc_template, parse_chroma_resposes
from magrathea.madeline.display import generate_md
from magrathea.madeline.category_embedder import get_mns_category

product_description_prompt = PromptTemplate(
    input_variables=["chat_history", "product_search"],
    template=product_desc_template,
)
memory = ConversationBufferMemory(memory_key="chat_history")


def vector_lookup(collection, response: dict, api=False, location="products_recs.md"):
    # call vector databses
    results = []
    for category, product_desc in response.items():
        try:
            result = collection.query(
                query_texts=[product_desc],
                n_results=1,
                include=["metadatas"],
                where={"category": get_mns_category(category)},
            )
            results.append(result)

        except NoDatapointsException:
            print(f"No products found for {category=}, removing category restriction")
            result = collection.query(query_texts=[product_desc], n_results=1, include=["metadatas"])
            results.append(result)

    # get product metadatas
    product_metadatas: list = parse_chroma_resposes(results)

    if not api:
        generate_md(product_metadatas, save_md=True, location=location)
    else:
        return product_metadatas


def generate_response(user_input, chain: LLMChain):
    result = chain.predict(product_search=user_input)
    return result
