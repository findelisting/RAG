import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
#from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core import StorageContext
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import chromadb
from llama_index.core import Settings
import os
import json
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.llms.ollama import Ollama

# Initialize the Chroma Client and Collection
# This is where the chroma.sqlite3 file is created
# Configure the Vector Store and Storage Context

chroma_client = chromadb.PersistentClient(path="./vector_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


#creating a json that contains career info


# Load structured dataset
json_path = r"C:\Users\ReDI User\Desktop\new\career_paths.json"
with open(json_path, "r", encoding="utf-8") as f:
    career_data = json.load(f)

def get_career_info(career: str, level: str = None) -> str:
    """Direct lookup from the career dataset."""
    c = career_data.get(career)
    if not c:
        return f"No data for {career}."
    if level:
        lvl = c["levels"].get(level)
        return json.dumps(lvl, indent=2) if lvl else f"No data for {level} level."
    return json.dumps(c, indent=2)


# Settings control global defaults
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Point the Ollama LLM to your desktop server
llm = Ollama(
    model="gpt-oss:20b",  # your model on desktop
    base_url="http://192.168.1.102:11434",  # desktop LAN IP
    request_timeout=360.0,
    context_window=8000,
)




Settings.llm = llm

#Creating a Rag tool using llamaindex
documents = SimpleDirectoryReader("data").load_data()

             
#Creating the chunks
parser = LangchainNodeParser (RecursiveCharacterTextSplitter (chunk_size=800, chunk_overlap = 0))
nodes = parser.get_nodes_from_documents (documents)
nodes [0:1]


index = VectorStoreIndex.from_documents (
    nodes,
    embed_model=embed_model, 
   
)
#to persist after setting teh document you save it to disk
index.storage_context.persist("storage")

query_engine = index.as_query_engine (
    #we can optionally override the llm here
    #llm=Settings.llm,
)

async def search_documents (query:str) -> str:
    """ useful for answering natural language questions about a the career paths of users"""
    response = await query_engine.aquery (query)
    return str (response)



# Create the agent
agent = AgentWorkflow.from_tools_or_functions(
    [search_documents, get_career_info],
    llm=llm,
    system_prompt="""You are a helpful assistant that can perform search documents and fetch structured career info""")

# create context
ctx = Context(agent)



async def main():
    #response = await agent.run("what does a data scientist do", ctx=ctx)
    #print("Introduction response:", response)
    response = await agent.run("select three positions and draft a 3 skills required for the position", ctx=ctx)
    print("Recall name response:", response)



# Run the agent
if __name__ == "__main__":
    asyncio.run(main())



