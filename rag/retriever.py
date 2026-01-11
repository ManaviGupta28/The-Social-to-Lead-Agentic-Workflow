"""
RAG Retriever for AutoStream Knowledge Base

This module implements the RAG (Retrieval-Augmented Generation) pipeline:
1. Loads knowledge base from JSON
2. Creates document chunks with metadata
3. Builds FAISS vector store using Google embeddings
4. Provides semantic search functionality
"""

import json
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load environment variables

# Use local HuggingFace embeddings instead of Google (no API needed, completely free)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeBaseRetriever:
    """RAG retriever for AutoStream product knowledge"""
    
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the retriever with knowledge base
        
        Args:
            knowledge_base_path: Path to knowledge_base.json file
        """
        if knowledge_base_path is None:
            # Default to the rag directory
            current_dir = Path(__file__).parent
            knowledge_base_path = current_dir / "knowledge_base.json"
        
        self.knowledge_base_path = knowledge_base_path
        self.vectorstore = None
        
        # Use local HuggingFace embeddings (completely free, no API needed)
        # This model runs locally on your machine
        # We specify a local cache_folder to avoid issues with spaces in user's home directory
        cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Initializing HuggingFaceEmbeddings with cache at: {cache_dir}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Fast, lightweight model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=cache_dir
        )
        
        # Initialize the vector store
        self._build_vectorstore()
    
    def _load_knowledge_base(self) -> Dict:
        """Load knowledge base from JSON file"""
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_documents(self) -> List[Document]:
        """
        Convert knowledge base into LangChain documents
        
        Returns:
            List of Document objects with content and metadata
        """
        kb = self._load_knowledge_base()
        documents = []
        
        # Company overview
        documents.append(Document(
            page_content=f"{kb['company']}: {kb['description']}",
            metadata={"type": "company_info"}
        ))
        
        # Pricing plans
        for plan in kb['pricing_plans']:
            plan_text = f"{plan['name']} - {plan['price']}\n\n"
            plan_text += "Features:\n" + "\n".join(f"- {feature}" for feature in plan['features'])
            
            if plan.get('limitations'):
                plan_text += "\n\nLimitations:\n" + "\n".join(f"- {lim}" for lim in plan['limitations'])
            
            if plan.get('recommended_for'):
                plan_text += f"\n\nRecommended for: {plan['recommended_for']}"
            
            documents.append(Document(
                page_content=plan_text,
                metadata={
                    "type": "pricing_plan",
                    "plan_name": plan['name'],
                    "price": plan['price']
                }
            ))
        
        # Policies
        for policy in kb['policies']:
            policy_text = f"{policy['title']}\n\n{policy['description']}"
            documents.append(Document(
                page_content=policy_text,
                metadata={
                    "type": "policy",
                    "title": policy['title']
                }
            ))
        
        # FAQ
        for faq in kb['faq']:
            faq_text = f"Q: {faq['question']}\n\nA: {faq['answer']}"
            documents.append(Document(
                page_content=faq_text,
                metadata={
                    "type": "faq",
                    "question": faq['question']
                }
            ))
        
        return documents
    
    def _build_vectorstore(self):
        """Build FAISS vector store from documents"""
        print("🔨 [DEBUG] Building vector store...")
        documents = self._create_documents()
        print(f"📄 [DEBUG] Created {len(documents)} base documents.")
        
        # Split documents if they're too long
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"✂️ [DEBUG] Split into {len(split_docs)} chunks.")
        
        # Create FAISS vector store
        print("🏗️ [DEBUG] Creating FAISS index from chunks (this involves embedding)...")
        self.vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=self.embeddings
        )
        print("✅ [DEBUG] Vector store build complete.")
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User's question or search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            print("⚠️ [DEBUG] Vector store is None, initialization failed?")
            raise ValueError("Vector store not initialized")
        
        print(f"🔍 [DEBUG] Searching for: {query}")
        results = self.vectorstore.similarity_search(query, k=k)
        print(f"📄 [DEBUG] Found {len(results)} relevant documents.")
        return results
    
    def get_context(self, query: str, k: int = 3) -> str:
        """
        Get formatted context string for RAG
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        docs = self.retrieve(query, k=k)
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Context {i}]\n{doc.page_content}")
        
        context_str = "\n\n".join(context_parts)
        return context_str


# Singleton instance for reuse
_retriever_instance = None

def get_retriever() -> KnowledgeBaseRetriever:
    """Get or create the knowledge base retriever instance"""
    global _retriever_instance
    print("📍 [DEBUG] get_retriever() called")
    if _retriever_instance is None:
        print("📍 [DEBUG] Initializing KnowledgeBaseRetriever singleton...")
        _retriever_instance = KnowledgeBaseRetriever()
    return _retriever_instance
