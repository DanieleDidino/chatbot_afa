from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index import StorageContext
from llama_index.llms import OpenAI
from llama_index import OpenAIEmbedding
from llama_index.node_parser import SimpleNodeParser

from utils import param_dict

import openai
import environ

env = environ.Env()
environ.Env.read_env()
API_KEY = env('OPENAI_API_KEY')
openai.api_key = API_KEY

# Set paths
docs_path = "documents_pdf" # where the documents are
embedding_path = "vector_db" # where the vector database is saved

# Select LLM
llm= OpenAI(temperature=0, model_name="gpt-3.5-turbo")
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

CHUNK_SIZE = param_dict["chunk_size"]

def ask_overwrite():
    prompt_string = "Create a new vector database? This will OVERWRITE the old vector database. (Y/N)"
    overwrite = input(prompt_string)
    return overwrite == "Y"


def load_docs(doc_path):
    docs = SimpleDirectoryReader(input_dir=doc_path).load_data()
    return docs


def create_vector_db(docs, vector_db_folder, llm, embed_model, chunk_size):
    """
    Build an index (vector database using the VectorStoreIndex class of LlamaIndex).

    Args:
        docs (Document): An object of the Document class from LlamaIndex.
        vector_db_folder (str): Folder where to save the index.
        llm: OpenAI Chat large language models API.
            Example: llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        embed_model: OpenAI embedding models.
            Example: llm_emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        chunk_size (Int): chunck size.
        chunk_overlap (Int): = sentence chunk overlap (default=0).

    Returns:
        index (VectorStoreIndex): An object of the VectorStoreIndex class from 
            LlamaIndex to use to build a query engine.
    """

    # ----------------------------------------------------------------------------------
    # Define SimpleNodeParser: a node parser used to split a document loaded from a file
    # into Nodes (automatically detects the NodeParser to use based on file type).
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size,
        #chunk_overlap=chunk_overlap,
    )
    nodes = node_parser.get_nodes_from_documents(docs)
    
    # ----------------------------------------------------------------------------------
    # Define The ServiceContext: A bundle of commonly used resources used during
    # the indexing and querying stage in a LlamaIndex pipeline/application.
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    
    # ----------------------------------------------------------------------------------
    # Storage context: The storage context container is a utility container for storing 
    # nodes, indices, and vectors. It contains the following:
    # - docstore: BaseDocumentStore
    # - index_store: BaseIndexStore
    # - vector_store: VectorStore
    # - graph_store: GraphStore
    storage_context = StorageContext.from_defaults()

    # ----------------------------------------------------------------------------------
    # VectorStoreIndex: a data structure that allows for the retrieval of relevant context
    # for a user query. This is particularly useful for retrieval-augmented generation (RAG) use-cases.
    # VectorStoreIndex stores data in Node objects, which represent chunks of the original documents,
    # and exposes a Retriever interface that supports additional configuration and automation.
    print("Creating Vector Database ...")
    index = VectorStoreIndex(
        nodes=nodes,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True
    )

    print("Saving Vector Database ...")
    index.storage_context.persist(persist_dir=vector_db_folder)
    print("Done")
    
    return index


def save_vector_db(index, embedding_path):
    index.storage_context.persist(persist_dir=embedding_path)


if __name__ == "__main__":
    overwrite = ask_overwrite()

    if overwrite:
        docs = load_docs(docs_path)
        index = create_vector_db(
            docs=docs,
           vector_db_folder=embedding_path,
           llm=llm,
           embed_model=embed_model,
           chunk_size=CHUNK_SIZE
        )
        save_vector_db(index, embedding_path)
    else:
        print("NOT OVERWRITE...Vector Database not created")
