""" Qdrant Vector Database Operations """

from typing import List, Optional, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
import uuid
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """ Qdrant vector store manager """
    
    def __init__(self):
        logger.info(f"Connecting to Qdrant at {settings.qdrant_url}")
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.encoder = SentenceTransformer(settings.embedding_model)
        self._ensure_collections()
    
    def _ensure_collections(self):
        collections = [settings.blog_collection_name, settings.user_memory_collection_name]

        for collection_name in collections:
            if not self.client.collection_exists(collection_name):
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                if collection_name == settings.blog_collection_name:
                    logger.info("Creating payload indexes for blog_content collection")
                    try:
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="company_name",
                            field_schema=PayloadSchemaType.KEYWORD
                        )
                        logger.info("Created index for company_name")
                        
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name="blog_url",
                            field_schema=PayloadSchemaType.KEYWORD
                        )
                        logger.info("Created index for blog_url")
                    except Exception as e:
                        logger.warning(f"Could not create payload indexes: {e}")
    
    def add_blog_content(
            self, 
            content: str,
            blog_url: str,
            blog_title: str,
            company_name: str,
            chunk_size: int = 1000
    ) -> List[str]:
        """
        Add blog content to vector store

        Args:
            content: Blog text content
            blog_url: URL of the blog
            blog_title: Title of the blog
            company_name: Company name
            chunk_size: Characters per chunk
            
        Returns:
            List of point IDs
        """
        logger.info(f"Adding blog content to vector store: {blog_title}")

        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        logger.debug(f"Split content into {len(chunks)} chunks")

        embeddings = self.encoder.encode(chunks).tolist()

        points = []
        point_ids = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": chunk,
                    "blog_url": blog_url,
                    "blog_title": blog_title,
                    "company_name": company_name,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                }
            )

            points.append(point)
        
        self.client.upsert(
            collection_name=settings.blog_collection_name,
            points=points
        )

        logger.info(f"Successfully added {len(points)} chunks to vector store")
        return point_ids
    
    def search_blog_content(self, query: str, company_name: Optional[str] = None,
                            blog_url: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """
        Search for relevant blog content
        
        Args:
            query: Search query
            company_name: Optional company filter
            blog_url: Optional blog URL filter
            limit: Maximum results
            
        Returns:
            List of search results with content and metadata
        """
        logger.info(f"Searching vector store for: {query[:50]}... (company: {company_name}, url: {blog_url})")

        query_vector = self.encoder.encode(query).tolist()

        must_conditions = []
        if company_name:
            must_conditions.append(FieldCondition(
                key="company_name",
                match=MatchValue(value=company_name)
            ))
        if blog_url:
            must_conditions.append(FieldCondition(
                key="blog_url",
                match=MatchValue(value=blog_url)
            ))
        
        query_filter = Filter(must=must_conditions) if must_conditions else None
        results = self.client.query_points(
            collection_name=settings.blog_collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )

        formatted_results = []
        for result in results.points:
            payload = result.payload or {}
            formatted_results.append({
                "content": payload.get("content", ""),
                "blog_title": payload.get("blog_title", ""),
                "blog_url": payload.get("blog_url", ""),
                "company_name": payload.get("company_name", ""),
                "score": getattr(result, "score", None)
            })
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def check_blog_exists(self, blog_url: str) -> bool:
        """
        Check if blog content already exists in vector store
        """
        try:
            count_result = self.client.count(
                collection_name=settings.blog_collection_name,
                query_filter=Filter(
                    must=[FieldCondition(
                        key="blog_url",
                        match=MatchValue(value=blog_url)
                    )]
                ),
                exact=True 
            )

            exists = count_result.count > 0
            logger.debug(f"Blog {blog_url} exists in vector store: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking if blog exists: {e}")
            return False
    
vector_store = VectorStore()
