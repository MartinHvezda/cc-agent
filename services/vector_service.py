"""
Vector Database Service for Issue Embedding and Search
Uses Qdrant for storing and searching similar customer issues
"""

import os
import uuid
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import json

load_dotenv()

class VectorService:
    """Service for managing issue embeddings and similarity search"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        self.collection_name = "customer_issues"
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536
        
        # Initialize collection
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created Qdrant collection: {self.collection_name}")
                print("ℹ️ Empty collection created. Use 'python scripts/seed_data.py' to add sample data.")
            else:
                print(f"✅ Qdrant collection '{self.collection_name}' already exists")
                
        except Exception as e:
            print(f"❌ Error initializing Qdrant collection: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            raise
    
    def add_issue(self, title: str, description: str, key: str = None, 
                  status: str = "Done", resolved_by_queue: str = "IAM", 
                  comments: List[str] = None) -> str:
        """Add a new issue to the vector database using Jira-like structure"""
        try:
            # Generate key if not provided
            if not key:
                key = f"CS-{str(uuid.uuid4())[:8].upper()}"
            
            # Combine title and description for embedding
            embedding_text = f"{title}. {description}"
            embedding = self.create_embedding(embedding_text)
            
            # Generate unique ID for Qdrant
            issue_id = str(uuid.uuid4())
            
            # Create point with Jira-like structure
            point = PointStruct(
                id=issue_id,
                vector=embedding,
                payload={
                    "key": key,
                    "title": title,
                    "description": description,
                    "status": status,
                    "resolved_by_queue": resolved_by_queue,
                    "comments": comments or [],
                    "created_at": "2024-01-20T10:00:00Z",  # In real app, use datetime.now()
                    "updated_at": "2024-01-20T10:00:00Z"
                }
            )
            
            # Insert into Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            print(f"✅ Added issue to vector DB: {key} ({issue_id})")
            return issue_id
            
        except Exception as e:
            print(f"❌ Error adding issue: {e}")
            raise
    
    def search_similar_issues(self, query: str, limit: int = 5, 
                            min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar issues using vector similarity"""
        try:
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_similarity
            )
            
            # Format results with Jira-like structure
            similar_issues = []
            for result in search_results:
                similar_issues.append({
                    "id": result.id,
                    "key": result.payload["key"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "status": result.payload["status"],
                    "resolved_by_queue": result.payload["resolved_by_queue"],
                    "comments": result.payload.get("comments", []),
                    "similarity_score": result.score,
                    "created_at": result.payload.get("created_at"),
                    "updated_at": result.payload.get("updated_at")
                })
            
            return similar_issues
            
        except Exception as e:
            print(f"❌ Error searching similar issues: {e}")
            return []
    
    def get_queue_analysis(self, query: str) -> Dict[str, Any]:
        """Analyze which queues have handled similar issues (for CC supervisor decision making)"""
        similar_issues = self.search_similar_issues(query, limit=5)
        
        if not similar_issues:
            return {
                "queue_analysis": {},
                "total_similar_issues": 0,
                "analysis_confidence": 0.0
            }
        
        # Analyze queue patterns in similar issues
        queue_stats = {}
        total_similarity = 0
        
        for issue in similar_issues:
            queue = issue["resolved_by_queue"]
            similarity = issue["similarity_score"]
            
            if queue not in queue_stats:
                queue_stats[queue] = {
                    "count": 0, 
                    "total_similarity": 0,
                    "avg_similarity": 0,
                    "issues": []
                }
            
            queue_stats[queue]["count"] += 1
            queue_stats[queue]["total_similarity"] += similarity
            queue_stats[queue]["issues"].append({
                "key": issue["key"],
                "title": issue["title"],
                "similarity": similarity
            })
            total_similarity += similarity
        
        # Calculate averages and percentages
        for queue, stats in queue_stats.items():
            stats["avg_similarity"] = stats["total_similarity"] / stats["count"]
            stats["percentage"] = (stats["count"] / len(similar_issues)) * 100
        
        analysis_confidence = total_similarity / len(similar_issues) if similar_issues else 0.0
        
        return {
            "queue_analysis": queue_stats,
            "total_similar_issues": len(similar_issues),
            "analysis_confidence": analysis_confidence,
            "similar_issues": similar_issues
        }
    
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count
            }
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return {}

# Global instance
vector_service = None

def get_vector_service() -> VectorService:
    """Get or create vector service instance"""
    global vector_service
    if vector_service is None:
        vector_service = VectorService()
    return vector_service