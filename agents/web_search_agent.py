from typing import Dict, List, Optional, Any

from services.web_search_service import WebSearchService


class WebSearchAgent:
    """Agent for performing web searches for stock information."""
    
    def __init__(self, web_search_service: WebSearchService):
        """
        Initialize the web search agent.
        
        Args:
            web_search_service: Service for performing web searches
        """
        self.web_search_service = web_search_service
    
    async def search(self, query: str) -> Dict[str, str]:
        """
        Perform a web search and process the results.
        
        Args:
            query: Search query string
            
        Returns:
            Dict containing processed search results
        """
        # Perform search
        search_results = await self.web_search_service.search(query)
        
        # Process results
        news_summary = self._extract_news_summary(search_results)
        outlook = self._extract_outlook(search_results)
        
        # Additional industry chain related information extraction
        recent_orders = self._extract_recent_orders(search_results)
        supply_chain_info = self._extract_supply_chain_info(search_results)
        industry_chain = self._extract_industry_chain(search_results)
        company_role = self._extract_company_role(search_results)
        
        return {
            "news_summary": news_summary,
            "outlook": outlook,
            "recent_orders": recent_orders,
            "supply_chain_info": supply_chain_info,
            "industry_chain": industry_chain,
            "company_role": company_role,
            "raw_results": search_results
        }

    def _extract_recent_orders(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Extract and summarize recent order information from search results.
        
        Args:
            search_results: List of search result items
            
        Returns:
            Summarized recent orders text
        """
        # Extract snippets related to orders
        order_snippets = []
        
        for result in search_results:
            snippet = result.get("snippet", "").lower()
            if any(keyword in snippet for keyword in ["order", "contract", "deal", "agreement", "partnership"]):
                order_snippets.append(result.get("snippet", ""))
        
        # Combine into a summary
        if not order_snippets:
            return "No specific information about recent orders found. Check the company's latest financial reports or news releases."
            
        # Format as markdown
        orders = "### Recent Order Details\n\n"
        for snippet in order_snippets[:3]:  # Limit to top 3 most relevant snippets
            orders += f"- {snippet}\n\n"
        
        return orders

    def _extract_supply_chain_info(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Extract supply chain information from search results.
        
        Args:
            search_results: List of search result items
            
        Returns:
            Extracted supply chain information text
        """
        # Extract snippets related to supply chain
        chain_snippets = []
        
        for result in search_results:
            snippet = result.get("snippet", "").lower()
            if any(keyword in snippet for keyword in ["supply chain", "supplier", "vendor", "procurement", "downstream", "upstream"]):
                chain_snippets.append(result.get("snippet", ""))
        
        # Combine into a summary
        if not chain_snippets:
            return "No detailed supply chain information found. Check the company's annual reports or industry research reports for more information."
            
        # Format as markdown
        supply_chain = "### Supply Chain Analysis\n\n"
        for snippet in chain_snippets[:3]:  # Limit to top 3 most relevant snippets
            supply_chain += f"- {snippet}\n\n"
        
        return supply_chain

    def _extract_industry_chain(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Extract industry chain information from search results.
        
        Args:
            search_results: List of search result items
            
        Returns:
            Extracted industry chain information text
        """
        # Extract snippets related to industry chain
        industry_snippets = []
        
        for result in search_results:
            snippet = result.get("snippet", "").lower()
            if any(keyword in snippet for keyword in ["industry chain", "value chain", "ecosystem", "industry outlook", "market trend"]):
                industry_snippets.append(result.get("snippet", ""))
        
        # Combine into a summary
        if not industry_snippets:
            return "No detailed industry chain analysis found. Check industry research reports for more information."
            
        # Format as markdown
        industry = "### Industry Chain Analysis\n\n"
        for snippet in industry_snippets[:3]:  # Limit to top 3 most relevant snippets
            industry += f"- {snippet}\n\n"
        
        return industry

    def _extract_company_role(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Extract the company's role in the industry chain from search results.
        
        Args:
            search_results: List of search result items
            
        Returns:
            Extracted company role information text
        """
        # Extract snippets related to company role
        role_snippets = []
        
        for result in search_results:
            snippet = result.get("snippet", "").lower()
            if any(keyword in snippet for keyword in ["market position", "role in", "contributes to", "key player", "industry leader"]):
                role_snippets.append(result.get("snippet", ""))
        
        # Combine into a summary
        if not role_snippets:
            return "No detailed information found about the company's specific role in the industry chain. Check industry analysis reports."
            
        # Format as markdown
        role = "### Company's Role in the Industry Chain\n\n"
        for snippet in role_snippets[:3]:  # Limit to top 3 most relevant snippets
            role += f"- {snippet}\n\n"
        
        return role
    
    def _extract_news_summary(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Extract and summarize news information from search results.
        
        Args:
            search_results: List of search result items
            
        Returns:
            Summarized news text
        """
        # Extract snippets related to news
        news_snippets = []
        
        for result in search_results:
            snippet = result.get("snippet", "").lower()
            if any(keyword in snippet for keyword in ["news", "announce", "report", "update", "recent"]):
                news_snippets.append(result.get("snippet", ""))
        
        # Combine into a summary
        if not news_snippets:
            return "No recent news found. Check the company's press releases or news section for updates."
            
        # Format as markdown
        news = "### Recent News\n\n"
        for snippet in news_snippets[:3]:  # Limit to top 3 most relevant snippets
            news += f"- {snippet}\n\n"
        
        return news

    def _extract_outlook(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Extract and summarize future outlook information from search results.
        
        Args:
            search_results: List of search result items
            
        Returns:
            Summarized outlook text
        """
        # Extract snippets related to outlook
        outlook_snippets = []
        
        for result in search_results:
            snippet = result.get("snippet", "").lower()
            if any(keyword in snippet for keyword in ["outlook", "forecast", "future", "prediction", "guidance", "expect"]):
                outlook_snippets.append(result.get("snippet", ""))
        
        # Combine into a summary
        if not outlook_snippets:
            return "No specific outlook information found. Check the company's latest earnings call transcript or investor presentations."
            
        # Format as markdown
        outlook = "### Future Outlook\n\n"
        for snippet in outlook_snippets[:3]:  # Limit to top 3 most relevant snippets
            outlook += f"- {snippet}\n\n"
        
        return outlook