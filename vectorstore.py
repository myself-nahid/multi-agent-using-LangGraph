import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

client = chromadb.Client() 

email_collection = client.get_or_create_collection(
    name="user_emails",
    metadata={"hnsw:space": "cosine"} # cosine distance for semantic search
)

# Google's embedding model
embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def add_emails(emails: list[str], ids: list[str]):
    """Embeds and stores emails in ChromaDB."""
    email_collection.add(
        ids=ids,
        documents=emails,
    )
    print(f"Added {len(emails)} emails to the vector store.")

def search_emails(query: str) -> str:
    """Searches for emails in ChromaDB based on a query."""
    
    query_embedding = embedding_function.embed_query(query)
    
   
    results = email_collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    
    if not results or not results.get("documents"):
        return "No relevant emails found."
        
    return "\n\n".join(results["documents"][0])