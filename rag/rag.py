import os
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms import Ollama
from llama_index.embeddings import OllamaEmbedding
import qdrant_client
import requests
import json


class RAG:
    def __init__(self, config_file, llm=None):
        """
        Initialize RAG with configuration and optional LLM
        
        Args:
            config_file: Dictionary containing configuration
            llm: Optional pre-configured LLM instance
        """
        self.config = config_file
        
        # Use provided LLM or create a new Ollama instance
        if llm is not None:
            self.llm = llm
        else:
            self.llm = Ollama(
                model=self.config.get("llm_model", "mistral"),
                base_url=self.config.get("ollama_url", "http://localhost:11434")
            )
            
        # Create embedding model
        self.embed_model = OllamaEmbedding(
            model_name=self.config.get("embed_model", "nomic-embed-text"),
            base_url=self.config.get("ollama_url", "http://localhost:11434")
        )
            
        # Set up Qdrant client
        self.qdrant_client = qdrant_client.QdrantClient(
            url=self.config.get("qdrant_url", "http://localhost:6333")
        )

    def qdrant_index(self):
        """
        Connect to Qdrant and load the index
        """
        print("Attempting to load index from Qdrant...")
        
        try:
            # Get collection name
            collection_name = self.config.get("collection_name", "research_papers")
            
            # Check if collection exists via direct HTTP
            qdrant_url = self.config.get("qdrant_url", "http://localhost:6333")
            response = requests.get(f"{qdrant_url}/collections/{collection_name}")
            
            if response.status_code != 200:
                print(f"Collection '{collection_name}' not found in Qdrant! HTTP Status: {response.status_code}")
                print("Please run data ingestion first.")
                return None
            
            print(f"Collection '{collection_name}' exists in Qdrant.")
            
            # Create vector store 
            qdrant_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name
            )
            
            # Create contexts
            storage_context = StorageContext.from_defaults(
                vector_store=qdrant_vector_store
            )
            
            service_context = ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=self.embed_model,
                chunk_size=self.config.get("chunk_size", 1024)
            )
            
            # Load index
            print(f"Loading existing index from collection: {collection_name}")
            try:
                index = load_index_from_storage(
                    storage_context=storage_context,
                    service_context=service_context
                )
                print("Index loaded successfully!")
                return index
            except Exception as e:
                print(f"Error loading index: {str(e)}")
                print("Attempting to create a new index from vector store...")
                
                # Try creating new index
                index = VectorStoreIndex.from_vector_store(
                    vector_store=qdrant_vector_store,
                    service_context=service_context
                )
                if index:
                    print("Created new index successfully!")
                    return index
                return None
        
        except Exception as e:
            print(f"Error in qdrant_index: {str(e)}")
            return None