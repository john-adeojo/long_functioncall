from haystack.nodes import PreProcessor
from haystack.utils import convert_files_to_docs
from haystack.document_stores import FAISSDocumentStore
from sqlalchemy import create_engine
from haystack.nodes import EmbeddingRetriever

# pre-process docs 
def preprocess_docs(doc_dir):
    all_docs = convert_files_to_docs(dir_path=doc_dir)
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_respect_sentence_boundary=True,
        split_overlap=30, 
        split_length=100
    )
    docs = preprocessor.process(all_docs)
    print(f"n_files_input: {len(all_docs)}\nn_docs_output: {len(docs)}")
    return docs

# create FAISS
def vector_stores(docs):
    engine = create_engine('sqlite:///C:/Users/johna/anaconda3/envs/longfunctioncall_env/long_functioncall/database/database.db')  # change to your local directory
    try:
        # Attempt to drop the table
        engine.execute("DROP TABLE document")
    except Exception as e:
        # Catch any exceptions, likely due to the table not existing
        print(f"Exception occurred while trying to drop the table: {e}")
    
    document_store = FAISSDocumentStore(sql_url='sqlite:///C:/Users/johna/anaconda3/envs/longfunctioncall_env/long_functioncall/database/database.db', faiss_index_factory_str="Flat", embedding_dim=768) # change to your local directory
    document_store.write_documents(docs)
    
    return document_store

def generate_embeddings(document_store):
    retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-mpnet-base-v2"
    )
    document_store.update_embeddings(retriever)
    return retriever