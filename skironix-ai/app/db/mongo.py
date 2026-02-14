from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

# Initialize Mongo Client
mongo_client = MongoClient(settings.MONGO_URI)
db = mongo_client[settings.DB_NAME]
vector_collection = db[settings.VECTOR_COLLECTION]

# Initialize Embeddings (Gecko)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=settings.GOOGLE_API_KEY
)

vector_store = MongoDBAtlasVectorSearch(
    collection=vector_collection,
    embedding=embeddings,
    index_name="vector_index",  # You must create this in Atlas UI
    relevance_score_fn="cosine",
)
