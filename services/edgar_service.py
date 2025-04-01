import os
import re
import aiofiles
from typing import Dict, List, Optional, Any
import asyncio
from sec_edgar_downloader import Downloader

from agents.utils import extract_text_from_filing


class EdgarService:
    """Service for downloading and processing SEC EDGAR filings."""
    
    def __init__(self, name: str, email: str, output_dir: str = "./"):
        """
        Initialize the SEC EDGAR service.
        
        Args:
            name: Requester name for SEC EDGAR
            email: Requester email for SEC EDGAR
            output_dir: Directory to store downloaded filings
        """
        self.name = name
        self.email = email
        self.output_dir = output_dir
        self.downloader = Downloader(name, email, output_dir)
    
    async def download_filings(self, symbol: str, form_type: str = "10-K", years: int = 5) -> List[str]:
        """
        Download SEC filings for a given stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
            form_type: SEC form type (default: 10-K)
            years: Number of years of filings to download
            
        Returns:
            List of paths to downloaded filings
        """
        # Convert to a coroutine
        return await asyncio.to_thread(self._sync_download_filings, symbol, form_type, years)
    
    def _sync_download_filings(self, symbol: str, form_type: str, years: int) -> List[str]:
        """
        Synchronous version of download_filings.
        
        Args:
            symbol: Stock symbol
            form_type: SEC form type
            years: Number of years
            
        Returns:
            List of filing paths
        """
        # Check if files already exist
        sec_dir = os.path.join(self.output_dir, "sec-edgar-filings", symbol, form_type)
        if os.path.exists(sec_dir):
            # Count existing filings
            existing_filings = [os.path.join(sec_dir, d) for d in os.listdir(sec_dir) if os.path.isdir(os.path.join(sec_dir, d))]
            if len(existing_filings) >= years:
                return self._get_filing_paths(existing_filings)
        
        # Download filings
        date_str = f"{years} years ago"
        self.downloader.get(form_type, symbol, after=date_str, download_details=False)
        
        # Get filing directories
        filing_dirs = []
        if os.path.exists(sec_dir):
            filing_dirs = [os.path.join(sec_dir, d) for d in os.listdir(sec_dir) if os.path.isdir(os.path.join(sec_dir, d))]
        
        return self._get_filing_paths(filing_dirs)
    
    def _get_filing_paths(self, filing_dirs: List[str]) -> List[str]:
        """
        Get paths to filing documents from filing directories.
        
        Args:
            filing_dirs: List of filing directory paths
            
        Returns:
            List of filing document paths
        """
        filing_paths = []
        
        for dir_path in filing_dirs:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(('.htm', '.html', '.xml', '.xbrl')):
                        filing_paths.append(os.path.join(root, file))
        
        return filing_paths
    
    async def convert_filing_to_text(self, filing_path: str) -> str:
        """
        Convert a filing to plain text.
        
        Args:
            filing_path: Path to filing file
            
        Returns:
            Plain text extracted from filing
        """
        # Convert to a coroutine
        return await asyncio.to_thread(extract_text_from_filing, filing_path)