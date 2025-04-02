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
    
    

    def _generate_technical_summary(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """
        Generate technical analysis summary.
        
        Args:
            symbol: Stock symbol
            data: DataFrame containing stock price data
            indicators: Dictionary containing calculated technical indicators
            
        Returns:
            Formatted technical analysis summary text
        """
        # Extract key indicators
        current_price = indicators["current_price"]
        ma_20 = indicators["ma_20"]
        ma_50 = indicators["ma_50"]
        ma_200 = indicators["ma_200"]
        macd = indicators["macd"]
        signal = indicators["signal"]
        rsi = indicators["rsi"]
        bb_middle = indicators["bb_middle"]
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        
        # Calculate price changes
        df = data.copy()
        week_change = df['close'].pct_change(periods=1).iloc[-1] * 100
        month_change = df['close'].pct_change(periods=4).iloc[-1] * 100
        ytd_change = df['close'].pct_change(periods=52).iloc[-1] * 100
        
        # Generate trend analysis
        if current_price > ma_50 and ma_50 > ma_200:
            trend = "Bullish (Long-Term Uptrend)"
        elif current_price > ma_50 and ma_50 < ma_200:
            trend = "Cautiously Bullish (Potential Golden Cross)"
        elif current_price < ma_50 and ma_50 > ma_200:
            trend = "Cautiously Bearish (Potential Short-Term Correction)"
        else:
            trend = "Bearish (Long-Term Downtrend)"
        
        # Format the summary text
        summary = f"""## Technical Analysis Summary

        ### Price Action
        - **Current Price:** ${current_price:.2f}
        - **Price Change (Week):** {week_change:.2f}%
        - **Price Change (Month):** {month_change:.2f}%
        - **Price Change (YTD):** {ytd_change:.2f}%
        
        ### Moving Averages
        - **20-Day MA:** ${ma_20:.2f} ({'Above' if current_price > ma_20 else 'Below'} current price)
        - **50-Day MA:** ${ma_50:.2f} ({'Above' if current_price > ma_50 else 'Below'} current price)
        - **200-Day MA:** ${ma_200:.2f} ({'Above' if current_price > ma_200 else 'Below'} current price)
        
        ### Technical Indicators
        - **Overall Trend:** {trend}
        - **RSI (14):** {rsi:.2f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
        - **MACD:** {macd:.2f} ({'Bullish' if macd > signal else 'Bearish'} crossover with signal line at {signal:.2f})
        - **Bollinger Bands:**
        - Upper Band: ${bb_upper:.2f}
        - Middle Band: ${bb_middle:.2f}
        - Lower Band: ${bb_lower:.2f}
        - Position: {
                'Near Upper Band (Potential Resistance)' if abs(current_price - bb_upper) < 0.05 * bb_middle else
                'Near Lower Band (Potential Support)' if abs(current_price - bb_lower) < 0.05 * bb_middle else
                'Mid-Range'
            }
        
        ### Support and Resistance Levels
        - **Key Resistance Levels:** [to be filled]
        - **Key Support Levels:** [to be filled]
        
        ### Trading Volume Analysis
        - **Recent Volume vs Average:** {'Above average' if indicators.get('recent_volume', 0) > indicators.get('avg_volume', 0) else 'Below average'} trading volume
        - **Volume Trend:** [to be filled]
        
        ### Key Price Levels to Watch
        - **Potential Upside Target:** ${current_price * 1.1:.2f} (10% upside)
        - **Potential Downside Risk:** ${current_price * 0.9:.2f} (10% downside)
        - **Stop Loss Suggestion:** ${current_price * 0.85:.2f} (15% downside protection)
        """
        
        return summary
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for stock data.
        
        Args:
            data: DataFrame containing stock price data
            
        Returns:
            Dictionary containing calculated technical indicators
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Rename columns to standardized names
        column_mapping = {}
        for col in df.columns:
            if '1. open' in col or 'open' in col.lower():
                column_mapping[col] = 'open'
            elif '2. high' in col or 'high' in col.lower():
                column_mapping[col] = 'high'
            elif '3. low' in col or 'low' in col.lower():
                column_mapping[col] = 'low'
            elif '4. close' in col or 'close' in col.lower():
                column_mapping[col] = 'close'
            elif '5. volume' in col or 'volume' in col.lower():
                column_mapping[col] = 'volume'
        
        # Apply the rename if we found mappings
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Check if 'close' column exists
        if 'close' not in df.columns:
            # Print available columns for debugging
            print(f"Available columns: {list(df.columns)}")
            
            # Try to find a suitable column
            for col in df.columns:
                if 'close' in col.lower() or 'price' in col.lower():
                    df['close'] = df[col]
                    print(f"Using '{col}' as 'close' column")
                    break
            else:
                # If we still don't have a 'close' column, use the 4th column (typical location of close price)
                if len(df.columns) >= 4:
                    df['close'] = df.iloc[:, 3]
                    print(f"Using column at index 3 as 'close' column")
                else:
                    raise ValueError(f"Cannot find 'close' price column in data. Available columns: {list(df.columns)}")
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic calculations
        current_price = df['close'].iloc[-1]
        
        # Simple Moving Averages (SMA)
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['ma_200'] = df['close'].rolling(window=200).mean()
        
        # Get the most recent values for moving averages
        ma_20 = df['ma_20'].iloc[-1]
        ma_50 = df['ma_50'].iloc[-1]
        ma_200 = df['ma_200'].iloc[-1]
        
        # MACD (Moving Average Convergence Divergence)
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Get recent MACD values
        macd = df['macd'].iloc[-1]
        signal = df['signal'].iloc[-1]
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Get recent RSI value
        rsi = df['rsi'].iloc[-1]
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)
        
        # Get recent Bollinger Band values
        bb_middle = df['bb_middle'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        
        # Volume analysis
        avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
        recent_volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0
        
        # Price momentum (rate of change)
        df['roc_5'] = df['close'].pct_change(periods=5) * 100
        df['roc_20'] = df['close'].pct_change(periods=20) * 100
        
        roc_5 = df['roc_5'].iloc[-1]
        roc_20 = df['roc_20'].iloc[-1]
        
        # Return all indicators in a dictionary
        return {
            "current_price": current_price,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "macd": macd,
            "signal": signal,
            "rsi": rsi,
            "bb_middle": bb_middle,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "avg_volume": avg_volume,
            "recent_volume": recent_volume,
            "roc_5": roc_5,
            "roc_20": roc_20,
            "dataframe": df  # Include the dataframe with all calculated indicators
        }