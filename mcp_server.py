from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context, Image
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io

from agents.rag_agent import RAGAgent
from agents.web_search_agent import WebSearchAgent
from agents.stock_data_agent import StockDataAgent
from services.edgar_service import EdgarService
from services.pinecone_service import PineconeService
from services.web_search_service import WebSearchService
from services.alpha_vantage_service import AlphaVantageService

load_dotenv()  # Load environment variables from .env file

@dataclass
class AppContext:
    edgar_service: EdgarService
    pinecone_service: PineconeService
    web_search_service: WebSearchService
    alpha_vantage_service: AlphaVantageService
    rag_agent: RAGAgent
    web_search_agent: WebSearchAgent
    stock_data_agent: StockDataAgent


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize and manage application lifecycle with services and agents."""
    # Initialize services
    edgar_service = EdgarService(
        name="Sicheng Bao", 
        email="Jellysillyfish@gmail.com",
        output_dir="./"
    )
    
    pinecone_service = PineconeService(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    
    web_search_service = WebSearchService(
        api_key=os.getenv("SERPAPI_API_KEY")
    )
    
    alpha_vantage_service = AlphaVantageService(
        api_key=os.getenv("ALPHA_VANTAGE_API_KEY")
    )
    
    # Initialize agents
    rag_agent = RAGAgent(
        edgar_service=edgar_service,
        pinecone_service=pinecone_service,
        gemini_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    web_search_agent = WebSearchAgent(
        web_search_service=web_search_service
    )
    
    stock_data_agent = StockDataAgent(
        alpha_vantage_service=alpha_vantage_service
    )
    
    context = AppContext(
        edgar_service=edgar_service,
        pinecone_service=pinecone_service,
        web_search_service=web_search_service,
        alpha_vantage_service=alpha_vantage_service,
        rag_agent=rag_agent,
        web_search_agent=web_search_agent,
        stock_data_agent=stock_data_agent
    )
    
    try:
        yield context
    finally:
        # Cleanup resources
        await pinecone_service.disconnect()


# Create MCP server
mcp = FastMCP(
    "Stock Research Report Generator",
    lifespan=app_lifespan,
    dependencies=[
        "pandas", "numpy", "matplotlib", "seaborn", "plotly",
        "pinecone-client", "google-generativeai", "sec-edgar-downloader",
        "alpha_vantage", "python-dotenv", "beautifulsoup4", "lxml"
    ]
)


@mcp.tool()
async def generate_stock_report(symbol: str, ctx: Context) -> str:
    """
    Generate a comprehensive 20-30 page research report for the specified stock.
    The report includes data from SEC filings, web search results, and financial analysis.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
    
    Returns:
        A detailed report with fundamental analysis, technical analysis, and news.
    """
    app_ctx = ctx.request_context.lifespan_context
    
    # Report progress
    ctx.info(f"Starting research for {symbol}")
    await ctx.report_progress(0, 4)
    
    # Step 1: RAG - Process SEC filings
    ctx.info(f"Processing SEC filings for {symbol}")
    rag_results = await app_ctx.rag_agent.process_filings(symbol)
    await ctx.report_progress(1, 4)
    
    # Step 2: Web search for recent news and analysis
    ctx.info(f"Searching for recent news about {symbol}")
    web_results = await app_ctx.web_search_agent.search(
        f"{symbol} stock recent orders supply chain partners financial analysis"
    )
    await ctx.report_progress(2, 4)
    
    # Step 3: Stock data analysis
    ctx.info(f"Analyzing stock performance data for {symbol}")
    stock_analysis = await app_ctx.stock_data_agent.analyze(symbol)
    await ctx.report_progress(3, 4)
    
    # Step 4: Compile final report
    ctx.info(f"Compiling final research report for {symbol}")
    report = f"""# Comprehensive Research Report for {symbol}

## 1. Company and Industry Overview

### Company Introduction
{rag_results['company_overview']}

### Industry Analysis
{rag_results.get('industry_analysis', 'Industry analysis data is being processed.')}

### Company's Position in the Industry
{rag_results.get('competitive_position', 'Competitive position analysis is being processed.')}

### Upstream and Downstream Supply-Demand Analysis
{web_results.get('supply_chain_info', 'Supply chain information is being processed.')}

## 2. Industry Chain Analysis

### Recent Major Orders
{web_results.get('recent_orders', 'Recent order information is being processed.')}

### Position of Orders in the Industry Chain
{web_results.get('industry_chain', 'Industry chain analysis is being processed.')}

### Current Status and Future Outlook of the Industry Chain
{web_results.get('industry_outlook', 'Industry outlook analysis is being processed.')}

### Company's Role in the Industry Chain
{web_results.get('company_role', 'Company role analysis is being processed.')}

## 3. Earnings Forecast and Investment Recommendations

### Main Business Revenue Analysis
{rag_results.get('revenue_analysis', 'Revenue analysis is being processed.')}

### Future Performance Forecast
{stock_analysis.get('earnings_forecast', 'Performance forecast is being processed.')}

### Valuation Analysis
{stock_analysis.get('valuation_analysis', 'Valuation analysis is being processed.')}

### Investment Recommendation
{stock_analysis.get('investment_recommendation', 'Investment recommendation is being processed.')}

## 4. Risk Assessment
{rag_results.get('risk_assessment', 'Risk assessment is being processed.')}

## Appendix: Data Sources
- SEC Edgar 10-K Reports (Last 5 years)
- Alpha Vantage Stock Price Data (Last 5 years, weekly data)
- Web Search Results (as of report generation date)
"""
    
    await ctx.report_progress(4, 4)
    ctx.info(f"Research report for {symbol} completed")
    
    return report

@mcp.tool()
async def generate_stock_analysis_chart(symbol: str, ctx: Context) -> Image:
    """
    Generate a stock analysis chart with price history, volume, and technical indicators.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
    
    Returns:
        An image of the stock price chart with technical indicators.
    """
    app_ctx = ctx.request_context.lifespan_context
    
    # Get stock data
    ctx.info(f"Retrieving historical data for {symbol}")
    stock_data = await app_ctx.alpha_vantage_service.get_weekly_data(symbol, years=5)
    
    # Create the chart
    ctx.info(f"Creating stock analysis chart for {symbol}")
    
    # Create figure with secondary Y axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.1,
                         subplot_titles=(f'{symbol} Stock Price', 'Volume'),
                         row_heights=[0.7, 0.3])
    
    # Add price data
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['1. open'],
            high=stock_data['2. high'],
            low=stock_data['3. low'],
            close=stock_data['4. close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['5. volume'],
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Add moving averages
    stock_data['MA50'] = stock_data['4. close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['4. close'].rolling(window=200).mean()
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['MA50'],
            name="50-day MA",
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['MA200'],
            name="200-day MA",
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} 5-Year Stock Analysis',
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    # Convert to image
    img_bytes = fig.to_image(format="png", width=1200, height=800)
    
    return Image(data=img_bytes, format="png")


@mcp.tool()
async def get_key_financial_ratios(symbol: str, ctx: Context) -> str:
    """
    Get key financial ratios and metrics for the specified stock.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
    
    Returns:
        A formatted table of key financial ratios.
    """
    app_ctx = ctx.request_context.lifespan_context
    
    # Get data from Alpha Vantage
    ctx.info(f"Retrieving financial metrics for {symbol}")
    overview = await app_ctx.alpha_vantage_service.get_company_overview(symbol)
    
    # Format data as a table
    ratios = [
        ("Market Cap", overview.get("MarketCapitalization", "N/A")),
        ("P/E Ratio", overview.get("PERatio", "N/A")),
        ("PEG Ratio", overview.get("PEGRatio", "N/A")),
        ("Price to Book", overview.get("PriceToBookRatio", "N/A")),
        ("Price to Sales", overview.get("PriceToSalesRatioTTM", "N/A")),
        ("EPS", overview.get("EPS", "N/A")),
        ("ROE", overview.get("ReturnOnEquityTTM", "N/A")),
        ("Profit Margin", overview.get("ProfitMargin", "N/A")),
        ("Operating Margin", overview.get("OperatingMarginTTM", "N/A")),
        ("Dividend Yield", overview.get("DividendYield", "N/A")),
        ("Dividend Per Share", overview.get("DividendPerShare", "N/A")),
        ("Beta", overview.get("Beta", "N/A")),
        ("52 Week High", overview.get("52WeekHigh", "N/A")),
        ("52 Week Low", overview.get("52WeekLow", "N/A"))
    ]
    
    # Format as markdown table
    result = f"# Key Financial Ratios for {symbol}\n\n"
    result += "| Metric | Value |\n"
    result += "|--------|-------|\n"
    
    for metric, value in ratios:
        result += f"| {metric} | {value} |\n"
    
    return result


@mcp.resource("company://{symbol}/overview")
async def get_company_overview(symbol: str, ctx: Context) -> str:
    """
    Get a comprehensive overview of the company based on its SEC filings.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
    """
    app_ctx = ctx.request_context.lifespan_context
    
    # Get data from the RAG agent
    ctx.info(f"Retrieving company overview for {symbol}")
    rag_results = await app_ctx.rag_agent.get_company_overview(symbol)
    
    return rag_results


@mcp.resource("market://{symbol}/news")
async def get_market_news(symbol: str, ctx: Context) -> str:
    """
    Get recent market news about the specified company.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
    """
    app_ctx = ctx.request_context.lifespan_context
    
    # Get data from the web search agent
    ctx.info(f"Searching for market news about {symbol}")
    news = await app_ctx.web_search_agent.get_news(symbol)
    
    return news


@mcp.resource("financial://{symbol}/data")
async def get_financial_data(symbol: str, ctx: Context) -> str:
    """
    Get key financial data for the specified company.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
    """
    app_ctx = ctx.request_context.lifespan_context
    
    # Get data from the stock data agent
    ctx.info(f"Retrieving financial data for {symbol}")
    financial_data = await app_ctx.stock_data_agent.get_financial_data(symbol)
    
    return financial_data


@mcp.prompt()
async def research_stock(symbol: str) -> str:
    """
    Create a prompt for researching a specific stock.
    
    Args:
        symbol: Stock symbol to research
    """
    return f"""
    Please create a comprehensive research report for {symbol} stock. This should include:
    
    1. A summary of the company's business model and industry position
    2. Key financial metrics and performance indicators
    3. Recent news and market developments affecting the company
    4. Technical analysis of stock price trends
    5. Growth prospects and potential risks
    6. Investment recommendation with supporting rationale
    
    Use all available data sources including SEC filings, market news, and stock price history.
    """


if __name__ == "__main__":
    mcp.run()