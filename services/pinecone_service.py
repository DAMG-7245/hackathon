import os
from typing import Dict, List, Optional, Any
import numpy as np
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec  # 更新导入语句


class PineconeService:
    """Service for interacting with Pinecone vector database."""
    
    def __init__(self, api_key: str, environment: str):
        """
        Initialize the Pinecone service.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
        """
        self.api_key = api_key
        self.environment = environment
        self.initialized = False
        self.embedding_model = genai.GenerativeModel('embedding-001')
        self.pc = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the Pinecone client."""
        if not self.initialized:
            # 使用新的API初始化
            self.pc = Pinecone(api_key=self.api_key)
            self.initialized = True
    
    async def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists in Pinecone.
        
        Args:
            index_name: Name of the index
            
        Returns:
            True if index exists, False otherwise
        """
        self._initialize()
        return index_name in self.pc.list_indexes().names()
    
    async def create_index(self, index_name: str, dimension: int = 768) -> None:
        """
        Create a new index in Pinecone.
        
        Args:
            index_name: Name of the index
            dimension: Dimension of the embeddings
        """
        self._initialize()
        
        if not await self.index_exists(index_name):
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
    
    async def delete_index(self, index_name: str) -> None:
        """
        Delete an index from Pinecone.
        
        Args:
            index_name: Name of the index
        """
        self._initialize()
        
        if await self.index_exists(index_name):
            self.pc.delete_index(index_name)
    
    async def index_documents(self, index_name: str, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents in Pinecone.
        
        Args:
            index_name: Name of the index
            documents: List of documents with text and metadata
        """
        self._initialize()
        
        if not await self.index_exists(index_name):
            await self.create_index(index_name)
        
        index = self.pc.Index(index_name)
        
        # Process documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Generate embeddings for the batch
            embeddings = await self._generate_embeddings([doc["text"] for doc in batch])
            
            # Prepare vectors for indexing
            vectors = []
            for j, embedding in enumerate(embeddings):
                vectors.append({
                    "id": f"doc_{i+j}",
                    "values": embedding,
                    "metadata": batch[j].get("metadata", {})
                })
            
            # Upsert vectors to Pinecone
            index.upsert(vectors=vectors)
    
    async def query(self, index_name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query the Pinecone index for similar documents.
        
        Args:
            index_name: Name of the index
            query: Query string
            limit: Maximum number of results
            
        Returns:
            List of similar documents with text and metadata
        """
        self._initialize()
        
        if not await self.index_exists(index_name):
            return []
        
        index = self.pc.Index(index_name)
        
        # Generate embedding for the query
        query_embedding = await self._generate_embeddings([query])
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding[0],
            top_k=limit,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.get("matches", []):
            formatted_results.append({
                "id": match.get("id", ""),
                "score": match.get("score", 0),
                "metadata": match.get("metadata", {}),
                "text": match.get("metadata", {}).get("text", "")
            })
        
        return formatted_results
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            # Generate embedding
            embedding = self.embedding_model.embed_content(
                content=text,
                task_type="retrieval_document"
            )
            
            # Extract embedding values
            embedding_values = embedding.embedding
            
            embeddings.append(embedding_values)
        
        return embeddings
    
    async def disconnect(self) -> None:
        """Clean up resources."""
        # Pinecone 不需要显式断开连接，
        # 但这个方法是为了与其他服务保持一致
        pass