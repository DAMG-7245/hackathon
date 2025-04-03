import os
import json
from typing import Dict, List, Optional, Any
import httpx
import logging
logger = logging.getLogger('web_search_service')
logger.setLevel(logging.DEBUG)


class WebSearchService:
    """Service for performing web searches using SERPAPI."""
    
    def __init__(self, api_key: str):
        """
        Initialize the web search service.
        
        Args:
            api_key: SERPAPI API key
        """
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        print(f"[DEBUG] 進入 WebSearchService.search 方法，查詢: {query}")
        """
        Perform a web search.
        
        Args:
            query: Search query string
            num_results: Number of results to retrieve
            
        Returns:
            List of search result items
        """
        # Prepare request parameters
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": num_results,
            "gl": "us",  # Google location parameter (United States)
            "hl": "en"   # Language (English)
        }
        
        # Make request to SERPAPI
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract organic results
                results = data.get("organic_results", [])
                
                # Format results
                formatted_results = []
                for result in results:
                    formatted_result = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "source": result.get("source", "")
                    }
                    formatted_results.append(formatted_result)
                
                return formatted_results
            else:
                print(f"Search API error: {response.status_code} - {response.text}")
                return []