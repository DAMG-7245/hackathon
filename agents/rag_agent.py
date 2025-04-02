import os
import time
import logging
from typing import Dict, List, Optional, Any

import google.generativeai as genai
from services.edgar_service import EdgarService
from services.pinecone_service import PineconeService

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_report_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('stock_report')

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
        logger.info("初始化RAGAgent")
        self.edgar_service = edgar_service
        self.pinecone_service = pinecone_service
        
        # Initialize Gemini
        logger.info("配置Gemini API")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("RAGAgent初始化完成")
    
    async def process_filings(self, symbol: str) -> Dict[str, str]:
        """
        Process SEC filings for a given stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
            
        Returns:
            Dict containing different sections of analyzed filing data
        """
        start_time = time.time()
        logger.info(f"开始处理股票 {symbol} 的文件")
        
        try:
            # 下载文件
            download_start = time.time()
            logger.info(f"开始下载 {symbol} 的SEC文件")
            filings = await self.edgar_service.download_filings(symbol)
            logger.info(f"下载完成，共 {len(filings)} 个文件，耗时 {time.time() - download_start:.2f} 秒")
            
            # 处理和索引文件
            index_start = time.time()
            logger.info(f"开始处理和索引 {symbol} 的文件")
            await self.process_and_index_filings(symbol, filings)
            logger.info(f"处理和索引完成，耗时 {time.time() - index_start:.2f} 秒")
            
            # 生成报告
            report_start = time.time()
            logger.info("开始生成报告部分")
            result = await self.generate_report_sections(symbol)
            logger.info(f"报告生成完成，耗时 {time.time() - report_start:.2f} 秒")
            
            logger.info(f"整个过程完成，总耗时 {time.time() - start_time:.2f} 秒")
            return result
        except Exception as e:
            logger.error(f"处理过程中出错: {str(e)}", exc_info=True)
            raise
    
    async def process_and_index_filings(self, symbol: str, filings: List[str]) -> None:
        """
        Process XBRL filings and index them in Pinecone.
        
        Args:
            symbol: Stock symbol
            filings: List of filing paths
        """
        symbol = symbol.lower()
        start_time = time.time()
        logger.info(f"开始处理和索引 {symbol} 的文件，共 {len(filings)} 个文件")
        
        try:
            # 检查索引是否存在
            index_check_start = time.time()
            logger.info(f"检查索引 {symbol} 是否存在")
            index_exists = await self.pinecone_service.index_exists(symbol)
            logger.info(f"检查索引存在耗时: {time.time() - index_check_start:.2f} 秒，结果: {index_exists}")
            
            if not index_exists:
                # 创建索引
                index_create_start = time.time()
                logger.info(f"开始为 {symbol} 创建索引")
                await self.pinecone_service.create_index(symbol)
                logger.info(f"创建索引完成，耗时 {time.time() - index_create_start:.2f} 秒")
                
                # 处理每个文件
                for i, filing_path in enumerate(filings):
                    file_start = time.time()
                    logger.info(f"开始处理第 {i+1}/{len(filings)} 个文件: {filing_path}")
                    
                    # 转换文件
                    convert_start = time.time()
                    logger.info(f"开始将文件转换为文本")
                    text = await self.edgar_service.convert_filing_to_text(filing_path)
                    logger.info(f"文件转换完成，耗时 {time.time() - convert_start:.2f} 秒，文本长度: {len(text)}")
                    
                    # 分块
                    chunk_start = time.time()
                    logger.info(f"开始将文本分块")
                    chunks = self._split_text_into_chunks(text)
                    logger.info(f"文本分块完成，耗时 {time.time() - chunk_start:.2f} 秒，共 {len(chunks)} 个块")
                    
                    # 索引块
                    index_docs_start = time.time()
                    logger.info(f"开始索引文档块")
                    await self.pinecone_service.index_documents(symbol, chunks)
                    logger.info(f"索引文档块完成，耗时 {time.time() - index_docs_start:.2f} 秒")
                    
                    logger.info(f"处理第 {i+1} 个文件完成，总耗时 {time.time() - file_start:.2f} 秒")
            else:
                logger.info(f"索引 {symbol} 已存在，跳过创建和索引步骤")
            
            logger.info(f"处理和索引 {symbol} 的所有文件完成，总耗时 {time.time() - start_time:.2f} 秒")
        except Exception as e:
            logger.error(f"处理和索引文件过程中出错: {str(e)}", exc_info=True)
            raise
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Split text into chunks for indexing.
        
        Args:
            text: The text to split
            chunk_size: Size of each chunk in characters
            
        Returns:
            List of document chunks with text and metadata
        """
        start_time = time.time()
        logger.info(f"开始将文本分块，文本长度: {len(text)}，块大小: {chunk_size}")
        
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
        
        logger.info(f"文本分块完成，生成了 {len(chunks)} 个块，耗时 {time.time() - start_time:.2f} 秒")
        return chunks
    
    async def generate_report_sections(self, symbol: str) -> Dict[str, str]:
        """
        Generate different sections of the report using RAG.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with different report sections
        """
        symbol = symbol.lower()
        start_time = time.time()
        logger.info(f"开始为 {symbol} 生成报告部分")
        
        try:
            # 定义要生成的部分
            section_queries = {
                "summary": "Provide a concise executive summary of the company.",
                "company_overview": "Describe the company's business model, products, and market position.",
                "industry_analysis": "Analyze the current state, size, growth trends, and major players in the company's industry.",
                "competitive_position": "Analyze the company's competitive position, market share, and competitive advantages in the industry.",
                "revenue_analysis": "Analyze the company's main business revenue composition, growth trends, and performance across business lines.",
                "risk_assessment": "Identify and analyze key risks mentioned in the company's SEC filings.",
                "detailed_financials": "Provide a detailed analysis of income statement, balance sheet, and cash flow statement."
            }
            
            # 生成每个部分
            sections = {}
            for section_name, query in section_queries.items():
                section_start = time.time()
                logger.info(f"开始生成 {section_name} 部分")
                sections[section_name] = await self._generate_section(symbol, query)
                logger.info(f"{section_name} 部分生成完成，耗时 {time.time() - section_start:.2f} 秒")
            
            logger.info(f"所有报告部分生成完成，总耗时 {time.time() - start_time:.2f} 秒")
            return sections
        except Exception as e:
            logger.error(f"生成报告部分时出错: {str(e)}", exc_info=True)
            raise
    
    async def _generate_section(self, symbol: str, query: str) -> str:
        """
        Generate a specific section of the report.
        
        Args:
            symbol: Stock symbol
            query: Query to run against the RAG system
            
        Returns:
            Generated text for the section
        """
        symbol = symbol.lower()
        start_time = time.time()
        logger.info(f"开始为 {symbol} 生成报告部分，查询: {query}")
        
        try:
            # 查询Pinecone
            query_start = time.time()
            logger.info(f"开始查询Pinecone索引")
            relevant_chunks = await self.pinecone_service.query(symbol, query, limit=5)
            logger.info(f"Pinecone查询完成，返回 {len(relevant_chunks)} 个结果，耗时 {time.time() - query_start:.2f} 秒")
            
            # 合并上下文
            context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
            logger.info(f"合并上下文完成，上下文长度: {len(context)}")
            
            # 生成响应
            generate_start = time.time()
            logger.info(f"开始使用Gemini生成内容")
            prompt = f"""
            Based on the following information from SEC filings for {symbol}:
            
            {context}
            
            {query}
            
            Provide a detailed, well-structured response focusing only on factual information from the filings.
            """
            
            response = self.model.generate_content(prompt)
            logger.info(f"Gemini内容生成完成，耗时 {time.time() - generate_start:.2f} 秒，响应长度: {len(response.text)}")
            
            logger.info(f"报告部分生成完成，总耗时 {time.time() - start_time:.2f} 秒")
            return response.text
        except Exception as e:
            logger.error(f"生成报告部分时出错: {str(e)}", exc_info=True)
            # 返回错误信息作为内容
            return f"Error generating section: {str(e)}"
    
    async def get_company_overview(self, symbol: str) -> str:
        """
        Get a comprehensive overview of the company.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company overview text
        """
        logger.info(f"获取公司概览: {symbol}")
        start_time = time.time()
        
        try:
            result = await self._generate_section(symbol, "Provide a comprehensive overview of the company, including its business model, products/services, market position, competitive advantages, and recent developments.")
            logger.info(f"公司概览生成完成，耗时 {time.time() - start_time:.2f} 秒")
            return result
        except Exception as e:
            logger.error(f"获取公司概览时出错: {str(e)}", exc_info=True)
            return f"Error generating company overview: {str(e)}"