from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import io

from services.alpha_vantage_service import AlphaVantageService


class StockDataAgent:
    """Agent for analyzing stock price data and financial metrics."""
    
    def __init__(self, alpha_vantage_service: AlphaVantageService):
        """
        Initialize the stock data agent.
        
        Args:
            alpha_vantage_service: Service for retrieving stock data
        """
        self.alpha_vantage_service = alpha_vantage_service
    
    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze stock data for the given symbol.
        
        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT, NVDA)
            
        Returns:
            Dict containing analysis results
        """
        # Get weekly stock data for the past 5 years
        stock_data = await self.alpha_vantage_service.get_weekly_data(symbol, years=5)
        
        # Calculate technical indicators
        tech_indicators = self._calculate_technical_indicators(stock_data)
        
        # Get company overview data
        overview = await self.alpha_vantage_service.get_company_overview(symbol)
        
        # Generate technical analysis summary
        technical_summary = self._generate_technical_summary(symbol, stock_data, tech_indicators)
        
        # Generate valuation analysis
        valuation_analysis = self._generate_valuation_analysis(symbol, overview, stock_data)
        
        # Generate earnings forecast
        earnings_forecast = self._generate_earnings_forecast(symbol, overview, stock_data)
        
        # Generate investment recommendation
        investment_recommendation = self._generate_investment_recommendation(
            symbol, overview, stock_data, tech_indicators, valuation_analysis
        )
        
        return {
            "stock_data": stock_data,
            "technical_indicators": tech_indicators,
            "company_overview": overview,
            "technical_summary": technical_summary,
            "valuation_analysis": valuation_analysis,
            "earnings_forecast": earnings_forecast,
            "investment_recommendation": investment_recommendation
        }

    def _generate_valuation_analysis(self, symbol: str, overview: Dict[str, Any], data: pd.DataFrame) -> str:
        """
        Generate valuation analysis.
        
        Args:
            symbol: Stock symbol
            overview: Company overview data
            data: DataFrame containing stock price data
            
        Returns:
            Formatted valuation analysis text
        """
        # Extract key financial metrics
        pe_ratio = overview.get("PERatio", "N/A")
        pb_ratio = overview.get("PriceToBookRatio", "N/A")
        ps_ratio = overview.get("PriceToSalesRatioTTM", "N/A")
        eps = overview.get("EPS", "N/A")
        
        # Generate valuation analysis text
        analysis = f"""## Valuation Analysis

    ### Current Valuation Metrics
    - **Price-to-Earnings (P/E):** {pe_ratio}
    - **Price-to-Book (P/B):** {pb_ratio}
    - **Price-to-Sales (P/S):** {ps_ratio}
    - **Earnings Per Share (EPS):** {eps}

    ### Industry Comparison
    Based on the above valuation metrics compared to the industry average:

    """
        
        # Add industry comparison analysis
        # This needs actual data, using example text for now
        if pe_ratio != "N/A" and float(pe_ratio) > 0:
            pe_value = float(pe_ratio)
            if pe_value > 30:
                analysis += "- **P/E Analysis:** Current P/E ratio is above the industry average, indicating high growth expectations from the market, but also potential overvaluation risk.\n"
            elif pe_value > 15:
                analysis += "- **P/E Analysis:** Current P/E ratio is at the industry average level, suggesting relatively reasonable valuation.\n"
            else:
                analysis += "- **P/E Analysis:** Current P/E ratio is below the industry average, potentially indicating undervaluation.\n"
        
        # Add DCF valuation analysis
        analysis += """
    ### Discounted Cash Flow (DCF) Valuation
    Based on historical growth rates, industry averages, and analyst expectations:

    - **5-year Revenue Forecast Growth Rate:** [to be filled]%
    - **Long-term Growth Rate:** [to be filled]%
    - **Weighted Average Cost of Capital (WACC):** [to be filled]%

    According to the DCF model, the company's reasonable valuation range is $[to be filled] - $[to be filled].

    ### Comprehensive Valuation Conclusion
    Combining multiple valuation methods, including relative valuation and absolute valuation models, [to be filled]
    """
        
        return analysis

    def _generate_earnings_forecast(self, symbol: str, overview: Dict[str, Any], data: pd.DataFrame) -> str:
        """
        Generate earnings forecast.
        
        Args:
            symbol: Stock symbol
            overview: Company overview data
            data: DataFrame containing stock price data
            
        Returns:
            Formatted earnings forecast text
        """
        # Extract company information
        sector = overview.get("Sector", "")
        industry = overview.get("Industry", "")
        
        # Generate earnings forecast text
        forecast = f"""## Earnings Forecast

    ### Main Business Revenue Analysis
    {symbol} belongs to the {industry} segment within the {sector} sector. Based on historical financial data analysis:

    - **Revenue Composition:** [to be filled]
    - **Main Growth Drivers:** [to be filled]
    - **Gross Margin Trends:** [to be filled]

    ### 3-Year Performance Forecast
    Based on industry trends, company historical performance, and market competition, the forecast is as follows:

    | Fiscal Year | Revenue(M) | YoY Growth | Net Profit(M) | YoY Growth | EPS | PE |
    |-------------|------------|------------|---------------|------------|-----|-----|
    | FY1         | [to be filled] | [to be filled]% | [to be filled] | [to be filled]% | [to be filled] | [to be filled] |
    | FY2         | [to be filled] | [to be filled]% | [to be filled] | [to be filled]% | [to be filled] | [to be filled] |
    | FY3         | [to be filled] | [to be filled]% | [to be filled] | [to be filled]% | [to be filled] | [to be filled] |

    ### Basis for Performance Forecast
    - **Industry Growth Forecast:** [to be filled]
    - **Expected Changes in Company Market Share:** [to be filled]
    - **New Product/Service Contribution:** [to be filled]
    - **Expected Cost Structure Changes:** [to be filled]
    """
        
        return forecast

    def _generate_investment_recommendation(
        self, 
        symbol: str, 
        overview: Dict[str, Any], 
        data: pd.DataFrame, 
        indicators: Dict[str, Any],
        valuation: str
    ) -> str:
        """
        Generate investment recommendation.
        
        Args:
            symbol: Stock symbol
            overview: Company overview data
            data: DataFrame containing stock price data
            indicators: Technical indicators dictionary
            valuation: Valuation analysis
            
        Returns:
            Formatted investment recommendation text
        """
        # Extract key metrics
        current_price = indicators.get("current_price", 0)
        
        # Signals based on technical analysis
        bullish_signals = 0
        bearish_signals = 0
        
        if indicators.get("ma_50", 0) > indicators.get("ma_200", 0): 
            bullish_signals += 1
        else: 
            bearish_signals += 1
        
        if indicators.get("macd", 0) > indicators.get("signal", 0): 
            bullish_signals += 1
        else: 
            bearish_signals += 1
        
        rsi = indicators.get("rsi", 50)
        if rsi > 50 and rsi < 70: 
            bullish_signals += 1
        elif rsi < 50 and rsi > 30: 
            bearish_signals += 1
        
        # Generate investment recommendation
        recommendation = f"""## Investment Recommendation

    """
        
        # Determine investment rating
        if bullish_signals > bearish_signals + 1:
            rating = "Buy"
            price_target = current_price * 1.15  # 15% upside
        elif bullish_signals > bearish_signals:
            rating = "Accumulate"
            price_target = current_price * 1.10  # 10% upside
        elif bearish_signals > bullish_signals + 1:
            rating = "Sell"
            price_target = current_price * 0.85  # 15% downside
        elif bearish_signals > bullish_signals:
            rating = "Reduce"
            price_target = current_price * 0.90  # 10% downside
        else:
            rating = "Hold"
            price_target = current_price * 1.05  # 5% upside
        
        recommendation += f"""### Investment Rating: {rating}
    ### Target Price: ${price_target:.2f} (Current Price: ${current_price:.2f})
    ### Potential Return: {((price_target/current_price - 1) * 100):.2f}%

    ### Rating Basis
    - **Fundamental Analysis:** [to be filled]
    - **Technical Analysis:** {'Bullish signals exceed bearish signals' if bullish_signals > bearish_signals else 'Bearish signals exceed bullish signals' if bearish_signals > bullish_signals else 'Balanced bullish and bearish signals'}
    - **Valuation Analysis:** [to be filled]
    - **Industry Outlook:** [to be filled]

    ### Investment Timeframe Recommendation
    - **Short-term (3-6 months):** [to be filled]
    - **Medium-term (6-18 months):** [to be filled]
    - **Long-term (18+ months):** [to be filled]

    ### Core Investment Logic
    [to be filled]
    """
        
        return recommendation