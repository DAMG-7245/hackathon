# 📊 Stock Report Generator (RAG + Financial Analysis)

---
## 🚀 Key Features

| Feature                    | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| 📄 RAG-based SEC Analysis | Uses Gemini + Pinecone to analyze SEC 10-K filings and generate report sections |
| 📈 Technical Analysis      | Calculates MA, MACD, RSI, Bollinger Bands, and other indicators             |
| 💰 Valuation & Forecast    | Combines PE/PB/EPS metrics for valuation + earnings forecast templates       |
| 📰 News & Supply Chain     | Scrapes recent news, industry outlook, and company supply chain role        |
| 📊 Visualization (Optional)| Generate Matplotlib charts for price and volume trends                     |

---

## 📁 Project Structure

```plaintext
HACKATHON/
├── agents/                        # Intelligent agents for analysis, querying, and content generation
│   ├── __init__.py
│   ├── rag_agent.py              # Uses Gemini + Pinecone to perform RAG-based SEC report analysis
│   ├── stock_data_agent.py       # Uses Alpha Vantage for technical and valuation analysis
│   ├── utils.py                  # Utilities for handling XBRL, HTML, and number formatting
│   └── web_search_agent.py       # Uses SERPAPI to fetch news and industry chain data
│
├── services/                     # Service layer for external APIs
│   ├── __init__.py
│   ├── alpha_vantage_service.py  # Retrieves fundamentals, price history, and financial statements
│   ├── edgar_service.py          # Downloads and parses SEC EDGAR filings into plain text
│   ├── pinecone_service.py       # Interfaces with Pinecone vector database
│   └── web_search_service.py     # Performs Google web search via SERPAPI
│
├── data/
│   └── data_processor.py         # Computes technical indicators, financial ratios, and generates charts
│
├── prompt/
│   └── report_templates.py       # Stores prompt templates used for LLM report generation
│
├── .env                          # Environment variables (e.g., API keys)
├── .gitignore                    # Files and directories to exclude from version control
├── mcp_server.py                 # Main program (can be CLI entry point or backend server)
```

---

## 🔧 Installation & Setup

### 1. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setup .env file
Create a .env file in the root directory with the following content:
```bash
# Alpha Vantage
ALPHA_VANTAGE_API_KEY=your_alpha_key

# Google Gemini
GEMINI_API_KEY=your_gemini_key

# SERPAPI (Web Search)
SERPAPI_API_KEY=your_serpapi_key

# SEC EDGAR
EDGAR_USER_NAME=YourName
EDGAR_USER_EMAIL=youremail@example.com

# Pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env

```

---

## **👨‍💻 Authors**
* Sicheng Bao (@Jellysillyfish13)
* Yung Rou Ko (@KoYungRou)
* Anuj Rajendraprasad Nene (@Neneanuj)

---

## **📞 Contact**
For questions, reach out via Big Data Course or open an issue on GitHub.