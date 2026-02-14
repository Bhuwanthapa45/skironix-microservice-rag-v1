from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.services.ingestion import get_retriever
from app.core.config import settings

# 1. Define Structured Output Schema
class Citation(BaseModel):
    text: str = Field(description="The specific regulation text or fact")
    source_page: int = Field(description="Page number")
    regulation_id: str = Field(description="The source document name")

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The comprehensive answer to the user's question")
    citations: list[Citation] = Field(description="List of exact citations used")

# 2. Setup Parser
parser = JsonOutputParser(pydantic_object=AnswerWithCitations)

# 3. Setup Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0,
    google_api_key=settings.GOOGLE_API_KEY
)

# 4. System Prompt
system_prompt_str = """
You are Skironix, a strict Aviation Regulation Compliance Officer.
Answer the user's question based ONLY on the context provided below.
If the answer is not in the context, state "I cannot find this in the uploaded regulations."

Format your output strictly as a JSON object with 'answer' and 'citations'.
Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_str),
    ("user", "{question}\n\n{format_instructions}")
])

def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'Unknown')} | Page: {d.metadata.get('page', 0)}] \nContent: {d.page_content}" for d in docs)

def get_rag_chain():
    retriever = get_retriever()
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough(), "format_instructions": lambda x: parser.get_format_instructions()}
        | prompt
        | llm
        | parser
    )
    return chain
