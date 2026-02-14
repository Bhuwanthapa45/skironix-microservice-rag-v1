from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.docstores import MongoDBDocStore
from pymongo import MongoClient
from app.core.config import settings
from app.db.mongo import embeddings # Import the Gecko embeddings we set up earlier

# 1. Connect to Mongo
client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]

# 2. Define the Collections
vector_collection = db[settings.VECTOR_COLLECTION] # 'regulation_vectors'
doc_collection = db[settings.DOC_STORE_COLLECTION] # 'regulation_store'

# 3. Setup Vector Store (For Child Chunks)
vector_store = MongoDBAtlasVectorSearch(
    collection=vector_collection,
    embedding=embeddings,
    index_name="vector_index",
    relevance_score_fn="cosine",
)

# 4. Setup Doc Store (For Parent Chunks)
# This replaces InMemoryStore. It saves the full text in 'regulation_store'
doc_store = MongoDBDocStore(
    collection=doc_collection,
)

def get_retriever():
    """
    Returns the Production Retriever.
    - Searches 'regulation_vectors' (Vector Search)
    - Retrieves full text from 'regulation_store' (Standard Mongo Query)
    """
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store, 
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever

async def process_pdf(file_path: str, source_id: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Tag documents so we can filter by them later (e.g. "Only search EASA docs")
    for doc in docs:
        doc.metadata["source_id"] = source_id
        doc.metadata["type"] = "regulation"

    retriever = get_retriever()
    
    # This magic line adds the small vectors to 'regulation_vectors' 
    # AND the big text to 'regulation_store' automatically.
    retriever.add_documents(docs, ids=None)
    
    return {"status": "success", "pages_processed": len(docs)}