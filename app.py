import chromadb
import os
from chromadb.config import Settings
from chromadb.errors import NoDatapointsException

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from magrathea.madeline.query import product_desc_template, parse_gpt_response
from magrathea.madeline.chatbot_rec import generate_response, vector_lookup
from magrathea.madeline.load_mns_articles import load_data

from datetime import datetime

from flask import Flask, request, jsonify

app = Flask(__name__)

product_description_prompt = PromptTemplate(
    input_variables=["chat_history", "product_search"],
    template=product_desc_template,
)

llm = OpenAI(temperature=0.1, max_tokens=1000, model_name="gpt-3.5-turbo")
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chroma/persist"))
collection = client.get_or_create_collection(name="test")

os.environ['OPENAI_API_KEY'] = "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"
os.environ['MONGO_STR'] = "mongodb+srv://tuesday-admin:SGSSfS5JsIx8BiwR@test.vek0t.azure.mongodb.net/product-service?retryWrites=true&w=majority&readPreference=secondary"


class Recommender:
    def __init__(self):
        memory = ConversationBufferMemory(memory_key="chat_history")
        self.chain = LLMChain(llm=llm, prompt=product_description_prompt, memory=memory, verbose=True)

    def reset(self):
        memory = ConversationBufferMemory(memory_key="chat_history")
        self.chain = LLMChain(llm=llm, prompt=product_description_prompt, memory=memory, verbose=True)


recommender = Recommender()

def get_recommended_outfits(user_input):
    result = generate_response(user_input, recommender.chain)
    response = parse_gpt_response(result)
    products = vector_lookup(collection, response, api=True)
    return products

@app.get("/recommendations")
def get_recommendation():
    user_input = request.args.get("user_input")
    return jsonify(get_recommended_outfits(user_input))

@app.get("/load_data")
def load_recomemendation_data():
    load_data(client)
    return 'complete'

@app.get("/refresh_session")
def refresh_session():
    recommender.reset()
    return 'Session refreshed'



@app.route('/')
def index():
   return "please hit the endpoint /hello"


@app.route('/hello', methods=['GET'])
def hello():
   # name = request.form.get('name')
    return "hello Sanjay"



if __name__ == '__main__':
    app.run()



