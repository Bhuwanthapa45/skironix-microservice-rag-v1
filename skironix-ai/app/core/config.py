from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    MONGO_URI: str
    DB_NAME: str
    VECTOR_COLLECTION: str
    DOC_STORE_COLLECTION: str
    API_SECRET_KEY: str

    class Config:
        env_file = ".env"

setting = Settings()
