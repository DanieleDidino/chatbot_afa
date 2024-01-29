from llama_index.prompts import PromptTemplate
from llama_index import StorageContext, load_index_from_storage
from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
import streamlit as st

param_dict = {
    "chunk_size": 256, # [256, 512, 1024]
    "top_k_results": 5 # [1, 3, 5]
}


def create_prompt_template():
    """
    Build a prompt template to use in query/chat engine.
    The template string must contain the expected parameters (e.g. {context_str} and {query_str}).

    Args: None

    Returns:
        A prompt template.
    """

    template = (
        "You are an expert Q&A system\n"
        "Keep your answers based on facts, do not hallucinate information.\n"
        "Always answer the query using the provided context information, and not prior knowledge.\n"
        "If an answer is not contained within the context information, print 'Sorry, not sufficient context information.'\n"
        "We have provided context information below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given this context information and not prior knowledge, please provide me with an answer to the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    return PromptTemplate(template)


def get_filename_from_title(my_dict, title):
    """
    Find a file_name (i.e., the key) in the dictionary given a title (i.e., the value).
    The dictionary has the following structure:
    {
        "file_name_1":"title_1",
        "file_name_2":"title_2",
        ...
    }

    Args:
        my_dict (dict): The dictionary with file names and titles.
        title (str): The title we want to use to find the file name (i.e., the key).

    Returns:
        query_engine: a query_engine created from the index.
    """
    keys_list = list(my_dict.keys())
    values_list = list(my_dict.values())
    return keys_list[values_list.index(title)]


def build_index_from_file(folder_with_index):
    """
    Rebuild storage context from a vector database and return a query engine.

    Args:
        folder_with_index (str): Folder where the vector database is.

    Returns:
        index (VectorStoreIndex): An object of the VectorStoreIndex class from 
            LlamaIndex to use to build a query engine.
    """
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=folder_with_index)
    # load index
    index = load_index_from_storage(storage_context)
    return index


def build_query_engine(index, prompt_template, param_dict):
    """
    Return a query engine.

    Args:
        index (VectorStoreIndex): An object of the VectorStoreIndex class from LlamaIndex.
        prompt_template (f-string): A prompt used to create the query engine.
        param_dict: A dictionary with the following key-value pairs:
            - "top_k_results" (int): Number of top results to return.

    Returns:
        query_engine: a query_engine created from the index.
    """
    query_engine = index.as_query_engine(
        text_qa_template=prompt_template,
        similarity_top_k=param_dict["top_k_results"]
    )
    return query_engine


def build_query_engine_hyde(query_engine):
    """
    Return a query_engine that uses we use Hypothetical Document Embeddings (HyDE)
        to generate a hypothetical document and use it for embedding lookup.

    Args:
        query_engine: A query_engine.

    Returns:
        query_engine_hyde: A query engine that uses HyDE to retrieve the documents.
    """
    hyde = HyDEQueryTransform(
        include_original=True
    )
    query_engine_hyde = TransformQueryEngine(query_engine, hyde)
    return query_engine_hyde


def response_from_query_engine(query_engine, prompt, pdf_dict):
    """
    Return the response from the query engine.

    Args:
        query_engine: A query engine.
        prompt (str): Prompt from the user.
        pdf_dict (dict): Dictionary with the title of the pdf files (our documents).

    Returns:
        response_for_user (str): response produced by the LLM.
    """
    # Response from llm
    response = query_engine.query(prompt)
    # Get response as string
    response_text = response.response
    # Create first part of the source section (i.e., section of the response message with source documents)
    response_metadata_message = f'There are {len(response.metadata)} source files.'
    # Loop over all documents used as source
    for i, meta_data in enumerate(response.metadata):
        # Extract title from dictionary with {"file_name":"title"}, given a file name
        document_title = pdf_dict[response.metadata[meta_data]["file_name"]]
        # Append the title, if title is not in the list of used sources
        if document_title not in st.session_state.list_file_download:
            st.session_state.list_file_download.append(document_title)
        # Update the source section with the source metadata
        metadata_source = f'<br>**Source {i+1}**'
        metadata_file_page = f'page {response.metadata[meta_data]["page_label"]} from file *{document_title}*'
        score = response.source_nodes[i].score
        metadata_score = f'[<font style="background-color: #dbe9e8">Cosine similarity: {score:.4f}</font>]'
        response_metadata_message += f'{metadata_source}: {metadata_file_page}. {metadata_score}'
    # Add response_metadata_message (i.e., source section) after the LLM response text
    response_for_user = (f"{response_text}<br><br>{response_metadata_message}")
    return response_for_user
