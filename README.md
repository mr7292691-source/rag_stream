# 📄 RAG PDF Field Extractor

A production-grade document field extraction system using **Google Gemini AI** and **FAISS Vector Search**.

## ✨ Features

- **📄 PDF Processing** - Upload and extract text from PDF documents
- **🔍 RAG-based Extraction** - Use retrieval-augmented generation for accurate field extraction
- **⚖️ Flow Comparison** - Compare Zero-shot vs RAG extraction methods
- **📊 Benchmarking** - Test extraction consistency and performance
- **🔧 Algorithm Comparison** - Compare chunking algorithms (Sliding Window vs Recursive)
- **💡 Confidence Scoring** - LLM-generated confidence with reasoning

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Configure API Key

Enter your Google Gemini API key in the sidebar.

## 📁 Project Structure

```
stream/
├── app.py                      # Main entry point (~120 lines)
├── config.py                   # Configuration and constants
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── core/                       # Core business logic
│   ├── __init__.py
│   ├── pdf_reader.py           # PDF reading utilities
│   ├── chunking.py             # Chunking algorithms
│   ├── embeddings.py           # Embedding generation
│   ├── faiss_index.py          # FAISS index operations
│   ├── retrieval.py            # RAG retrieval logic
│   └── extraction.py           # Field extraction logic
│
├── analysis/                   # Analysis & comparison modules
│   ├── __init__.py
│   ├── document_analyzer.py    # Document field analysis
│   ├── flow_comparison.py      # Zero-shot vs RAG comparison
│   ├── hallucination.py        # Hallucination scoring
│   └── benchmarking.py         # Performance benchmarking
│
├── ui/                         # Streamlit UI components
│   ├── __init__.py
│   ├── sidebar.py              # Sidebar configuration
│   ├── styles.py               # CSS and styling
│   ├── components.py           # Reusable UI components
│   └── tabs/                   # Individual tab pages
│       ├── __init__.py
│       ├── extraction_tab.py   # Document Extraction
│       ├── analysis_tab.py     # Document Analysis
│       ├── comparison_tab.py   # Flow Comparison
│       ├── benchmark_tab.py    # Benchmarking
│       ├── algorithm_tab.py    # Algorithm Comparison
│       └── settings_tab.py     # Settings
│
└── utils/                      # Utility modules
    ├── __init__.py
    ├── text_highlight.py       # Text highlighting
    └── rate_limiter.py         # API rate limiting
```

## 🛠️ Usage

### Document Extraction
1. Upload a PDF document
2. Configure chunking settings in the sidebar
3. Click "Build Index"
4. Ask questions in natural language

### Flow Comparison
1. Provide a master output (ground truth)
2. Run Zero-shot and/or RAG extraction
3. Compare accuracy and hallucination scores

### Benchmarking
1. Enter a query to benchmark
2. Optionally provide expected output
3. Run multiple iterations
4. Analyze consistency and performance

## 🔧 Configuration

Edit `config.py` to customize:
- Model names
- Default chunking parameters
- API settings
- UI constants

## 📝 License

MIT License
