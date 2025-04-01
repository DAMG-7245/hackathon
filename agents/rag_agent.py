import os
from typing import Dict, List, Optional, Any

import google.generativeai as genai
from services.edgar_service import EdgarService
from services.pinecone_service import PineconeService


class RAGAgent:
    """Agent for retrieving and processing SEC filings using RAG."""
    
    def __init__(self, edgar_service: EdgarService, pinecone_service: PineconeService, gemini_api_key: str):
        """
        Initialize the RAG agent.
        
        Args:
            edgar_service: Service for downloading SEC filings
            pinecone_service: Service for vector database operations
            gemini_api_key: API key for Google Gemini
        """
        self.edgar_service = edgar_service
        self.pinecone_service = pinecone_service
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    async def process_filings(self, symbol: str) -> Dict[str, str]:
        """
        Process SEC filings for a given stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
            
        Returns:
            Dict containing different sections of analyzed filing data
        """
        # Download filings if not already downloaded
        filings = await self.edgar_service.download_filings(symbol)
        
        # Process and index filings if not already indexed
        await self.process_and_index_filings(symbol, filings)
        
        # Query the index to generate report sections
        return await self.generate_report_sections(symbol)
    
    async def process_and_index_filings(self, symbol: str, filings: List[str]) -> None:
        """
        Process XBRL filings and index them in Pinecone.
        
        Args:
            symbol: Stock symbol
            filings: List of filing paths
        """
        # Check if index exists
        index_exists = await self.pinecone_service.index_exists(symbol)
        
        if not index_exists:
            # Create index
            await self.pinecone_service.create_index(symbol)
            
            # Process each filing
            for filing_path in filings:
                # Convert XBRL to text
                text = await self.edgar_service.convert_filing_to_text(filing_path)
                
                # Split into chunks
                chunks = self._split_text_into_chunks(text)
                
                # Index chunks
                await self.pinecone_service.index_documents(symbol, chunks)
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Split text into chunks for indexing.
        
        Args:
            text: The text to split
            chunk_size: Size of each chunk in characters
            
        Returns:
            List of document chunks with text and metadata
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            chunks.append({
                "text": chunk,
                "metadata": {
                    "chunk_id": i // chunk_size,
                    "source": "sec_filing"
                }
            })
        return chunks
    
    async def generate_report_sections(self, symbol: str) -> Dict[str, str]:
        """
        Generate different sections of the report using RAG.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with different report sections
        """
        sections = {
            "summary": await self._generate_section(symbol, "Provide a concise executive summary of the company."),
            "company_overview": await self._generate_section(symbol, "Describe the company's business model, products, and market position."),
            "industry_analysis": await self._generate_section(symbol, "Analyze the current state, size, growth trends, and major players in the company's industry."),
            "competitive_position": await self._generate_section(symbol, "Analyze the company's competitive position, market share, and competitive advantages in the industry."),
            "revenue_analysis": await self._generate_section(symbol, "Analyze the company's main business revenue composition, growth trends, and performance across business lines."),
            "risk_assessment": await self._generate_section(symbol, "Identify and analyze key risks mentioned in the company's SEC filings."),
            "detailed_financials": await self._generate_section(symbol, "Provide a detailed analysis of income statement, balance sheet, and cash flow statement.")
        }
        
        return sections
        
       
    
    async def _generate_section(self, symbol: str, query: str) -> str:
        """
        Generate a specific section of the report.
        
        Args:
            symbol: Stock symbol
            query: Query to run against the RAG system
            
        Returns:
            Generated text for the section
        """
        # Query Pinecone for relevant chunks
        relevant_chunks = await self.pinecone_service.query(symbol, query, limit=5)
        
        # Combine chunks into context
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Generate response using Gemini
        prompt = f"""
        Based on the following information from SEC filings for {symbol}:
        
        {context}
        
        {query}
        
        Provide a detailed, well-structured response focusing only on factual information from the filings.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    async def get_company_overview(self, symbol: str) -> str:
        """
        Get a comprehensive overview of the company.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company overview text
        """
        return await self._generate_section(symbol, "Provide a comprehensive overview of the company, including its business model, products/services, market position, competitive advantages, and recent developments.")