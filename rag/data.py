import os
import argparse
import yaml
import qdrant_client
from tqdm import tqdm
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms import Ollama
from llama_index.embeddings import OllamaEmbedding
from pathlib import Path


class Data:
    def __init__(self, config):
        self.config = config

    def _create_data_folder(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"Output folder created at {data_path}")
        else:
            print(f"Using existing folder at {data_path}")
            
    def _initialize_ollama(self):
        """Initialize Ollama models for embedding and LLM"""
        try:
            ollama_embed = OllamaEmbedding(
                model_name=self.config.get("embed_model", "nomic-embed-text"),
                base_url=self.config.get("ollama_url", "http://localhost:11434")
            )
            
            ollama_llm = Ollama(
                model=self.config.get("llm_model", "mistral"),
                base_url=self.config.get("ollama_url", "http://localhost:11434")
            )
            return ollama_embed, ollama_llm
        except Exception as e:
            print(f"Error initializing Ollama models: {str(e)}")
            return None, None
            
    def _initialize_qdrant(self):
        """Initialize Qdrant client and collection"""
        try:
            client = qdrant_client.QdrantClient(url=self.config.get("qdrant_url", "http://localhost:6333"))
            collection_name = self.config.get("collection_name", "research_papers")
            
            # Check if collection exists
            try:
                collections_response = client.get_collections()
                existing_collections = [c.name for c in collections_response.collections]
                if collection_name in existing_collections:
                    print(f"Collection {collection_name} already exists. Updating...")
                else:
                    print(f"Creating new collection: {collection_name}")
            except Exception as e:
                print(f"Error checking collections: {str(e)} - will attempt to create")
                
            return client, collection_name
        except Exception as e:
            print(f"Error connecting to Qdrant: {str(e)}")
            return None, None

    def process_local_pdfs(self, pdf_directory=None):
        """
        Process PDF files from a local directory.
        
        Args:
            pdf_directory (str): Path to directory containing PDF files.
                                If None, uses the path from config.
        
        Returns:
            str: Path to the directory with PDFs
        """
        pdf_path = pdf_directory or self.config["data_path"]
        self._create_data_folder(pdf_path)
        
        if not os.path.exists(pdf_path):
            print(f"Directory {pdf_path} does not exist.")
            return pdf_path
        
        pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_path}")
        else:
            print(f"Found {len(pdf_files)} PDF files in {pdf_path}")
            for pdf in tqdm(pdf_files, desc="Processing local PDFs"):
                print(f"Found: {pdf}")
        
        return pdf_path

    def ingest_single_pdf(self, pdf_file_path):
        """
        Ingest a single PDF file into the vector database.
        
        Args:
            pdf_file_path (str): Path to the PDF file to ingest
            
        Returns:
            VectorStoreIndex or None: The index if successful, None otherwise
        """
        if not os.path.exists(pdf_file_path):
            print(f"Error: File {pdf_file_path} does not exist.")
            return None
            
        if not pdf_file_path.endswith('.pdf'):
            print(f"Error: File {pdf_file_path} is not a PDF.")
            return None
        
        print(f"Indexing PDF: {pdf_file_path}")
        
        # Initialize Ollama
        print("Initializing Ollama models...")
        ollama_embed, ollama_llm = self._initialize_ollama()
        if not ollama_embed or not ollama_llm:
            return None
        
        # Load document
        print("Loading PDF document...")
        try:
            file_dir = str(Path(pdf_file_path).parent)
            file_name = Path(pdf_file_path).name
            documents = SimpleDirectoryReader(file_dir, file_names=[file_name]).load_data()
            print(f"Loaded PDF: {file_name}")
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return None
        
        # Connect to Qdrant
        print("Connecting to Qdrant...")
        client, collection_name = self._initialize_qdrant()
        if not client:
            return None
        
        try:
            # Create vector store
            qdrant_vector_store = QdrantVectorStore(
                client=client, collection_name=collection_name
            )
            
            # Create contexts
            storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
            service_context = ServiceContext.from_defaults(
                llm=ollama_llm, 
                embed_model=ollama_embed,
                chunk_size=self.config.get("chunk_size", 1024)
            )

            # Index document
            print("Indexing PDF (this may take a while)...")
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                service_context=service_context
            )
            
            print(f"PDF indexed successfully to Qdrant. Collection: {collection_name}")
            return index
        
        except Exception as e:
            print(f"Error during indexing: {str(e)}")
            return None

    def ingest(self, data_path=None):
        """
        Ingest documents into the vector database using local Ollama models.
        """
        doc_path = data_path or self.config["data_path"]
        print(f"Indexing data from {doc_path}...")
        
        # Check if directory exists and contains PDF files
        if not os.path.exists(doc_path):
            print(f"Error: Directory {doc_path} does not exist.")
            return None
            
        pdf_files = [f for f in os.listdir(doc_path) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"Error: No PDF files found in {doc_path}")
            return None
            
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize Ollama
        print("Initializing Ollama models...")
        ollama_embed, ollama_llm = self._initialize_ollama()
        if not ollama_embed or not ollama_llm:
            return None
        
        # Load documents
        print("Loading documents...")
        try:
            documents = SimpleDirectoryReader(doc_path).load_data()
            print(f"Loaded {len(documents)} documents")
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return None
        
        # Connect to Qdrant
        print("Connecting to Qdrant...")
        client, collection_name = self._initialize_qdrant()
        if not client:
            return None
        
        try:
            # Create vector store
            qdrant_vector_store = QdrantVectorStore(
                client=client, collection_name=collection_name
            )
            
            # Create contexts
            storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
            service_context = ServiceContext.from_defaults(
                llm=ollama_llm, 
                embed_model=ollama_embed,
                chunk_size=self.config.get("chunk_size", 1024)
            )

            # Index documents
            print("Indexing documents (this may take a while)...")
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                service_context=service_context
            )
            
            print(f"Data indexed successfully to Qdrant. Collection: {collection_name}")
            return index
        
        except Exception as e:
            print(f"Error during indexing: {str(e)}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and ingest local PDF files using fully local models")
    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=None,
        help="Path to local directory containing PDF files. Uses config path if not specified.",
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Path to a single PDF file to ingest.",
    )
    
    parser.add_argument(
        "-i", "--ingest",
        action="store_true",
        default=True,
        help="Ingest data to local Qdrant vector database. Default: True",
    )

    args = parser.parse_args()
    
    # Load configuration
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    
    # Create data processor
    data = Data(config)
    
    if args.file:
        # Ingest single PDF file
        data.ingest_single_pdf(args.file)
    else:
        # Process local PDFs
        data_path = data.process_local_pdfs(args.directory)
        
        # Perform ingestion if requested
        if args.ingest:
            data.ingest(data_path=data_path)