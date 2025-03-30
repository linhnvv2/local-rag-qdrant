from fastapi import FastAPI, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import yaml
import requests
import json
import logging
import os
from llama_index.llms import Ollama
from llama_index.embeddings import OllamaEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_file = "config.yml"
with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)

# Define model classes
class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)

class Response(BaseModel):
    search_result: str 
    source: str

# Direct Qdrant functions that bypass client validation issues
def check_collection_exists(collection_name: str, qdrant_url: str = "http://localhost:6333") -> bool:
    """Check if a collection exists in Qdrant using direct HTTP request"""
    try:
        response = requests.get(f"{qdrant_url}/collections/{collection_name}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking collection: {str(e)}")
        return False

def query_qdrant_directly(query_vector: List[float], collection_name: str, top_k: int = 1, 
                          qdrant_url: str = "http://localhost:6333") -> Optional[List[Dict[Any, Any]]]:
    """Query Qdrant directly using the REST API instead of the client library"""
    logger.info(f"Directly querying Qdrant collection {collection_name} via REST API")
    
    # Query endpoint
    url = f"{qdrant_url}/collections/{collection_name}/points/search"
    
    # Query payload
    payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False
    }
    
    try:
        # Make the request
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Error searching Qdrant: Status {response.status_code}, {response.text}")
            return None
            
        # Parse response
        result = response.json()
        return result.get("result", [])
    except Exception as e:
        logger.error(f"Exception querying Qdrant directly: {str(e)}")
        return None

# Initialize RAG components
def initialize_ollama_models():
    """Initialize Ollama LLM and embedding models"""
    try:
        ollama_url = config.get("ollama_url", "http://localhost:11434")
        
        llm = Ollama(
            model=config.get("llm_model", "mistral"),
            base_url=ollama_url
        )
        
        embed_model = OllamaEmbedding(
            model_name=config.get("embed_model", "nomic-embed-text"),
            base_url=ollama_url
        )
        
        return llm, embed_model
    except Exception as e:
        logger.error(f"Error initializing Ollama models: {str(e)}")
        return None, None

# Try to initialize Ollama models
llm, embed_model = initialize_ollama_models()

# Check if the collection exists
collection_name = config.get("collection_name", "research_papers")
qdrant_url = config.get("qdrant_url", "http://localhost:6333")
collection_exists = check_collection_exists(collection_name, qdrant_url)

if collection_exists:
    logger.info(f"Collection '{collection_name}' exists in Qdrant")
else:
    logger.warning(f"Collection '{collection_name}' not found in Qdrant - please ingest documents first")

# Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    if not collection_exists:
        return {"message": "Research RAG API (No index found - please ingest documents first)"}
    if not llm or not embed_model:
        return {"message": "Research RAG API (Error initializing Ollama models - check if Ollama is running)"}
    return {"message": "Research RAG API is ready for queries"}

@app.post("/api/search", response_model=Response)
async def search(query: Query):
    # Check if models are initialized
    if not llm or not embed_model:
        raise HTTPException(
            status_code=503,
            detail="Error initializing LLM or embedding model. Check if Ollama is running."
        )
    
    # Check if collection exists
    if not collection_exists:
        raise HTTPException(
            status_code=503,
            detail="No document collection found. Please run data ingestion first: python rag/data.py --directory './data'"
        )
    
    logger.info(f"Processing query: '{query.query}' with similarity_top_k={query.similarity_top_k}")
    
    try:
        # Generate embedding for the query
        query_embedding = embed_model.get_text_embedding(query.query)
        
        # Query Qdrant directly
        results = query_qdrant_directly(
            query_vector=query_embedding,
            collection_name=collection_name,
            top_k=query.similarity_top_k,
            qdrant_url=qdrant_url
        )
        
        if not results:
            return Response(
                search_result="I couldn't find any relevant information to answer your question in the research documents.",
                source="No matching documents found"
            )
        
        # Extract text from the results
        contexts = []
        sources = set()
        
        for result in results:
            if "payload" in result and "text" in result["payload"]:
                contexts.append(result["payload"]["text"])
                
                # Extract source if available
                try:
                    if "metadata" in result["payload"] and "file_path" in result["payload"]["metadata"]:
                        sources.add(result["payload"]["metadata"]["file_path"])
                except:
                    pass
        
        # Prepare context for LLM
        context_text = "\n\n".join(contexts)
        
        # Format prompt
        prompt = f"""Based on the following context from research papers, please answer the question. 
If the answer cannot be determined from the context, say 'I don't have enough information from the research papers to answer this question.'

Context:
{context_text}

Question: {query.query}

Answer:"""
        
        # Get response from LLM
        response_text = llm.complete(prompt).text
        
        # Format source for display
        source_text = ", ".join(sources) if sources else "Unknown source"
        
        # Return the response
        return Response(
            search_result=response_text.strip(),
            source=source_text
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )