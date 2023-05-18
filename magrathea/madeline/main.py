import chromadb
from chromadb.config import Settings
from chromadb.errors import NoDatapointsException

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from magrathea.madeline.query import product_desc_template, parse_gpt_response
from magrathea.madeline.chatbot_rec import generate_response, vector_lookup

from datetime import datetime

product_description_prompt = PromptTemplate(
    input_variables=["chat_history", "product_search"],
    template=product_desc_template,
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI(temperature=0.5, max_tokens=1000, model_name="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=product_description_prompt, memory=memory, verbose=True)
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chroma/persist"))
collection = client.get_or_create_collection(name="test")

if __name__ == "__main__":
    answer = 'n'
    i = 0
    current_datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    while answer == 'n':
        i += 1
        result = generate_response(chain)
        print(result)
        response = parse_gpt_response(result)
        vector_lookup(collection, response, f'data/{current_datetime_str}/products_recs{i}.md')
        print(result)
        answer = input('Are you happy with this? Enter y/n')

