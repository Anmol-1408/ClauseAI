# HackRX PDF Question Answering API

A FastAPI-based application that processes PDF documents and answers questions using AI. Built for the HackRX competition with vector search capabilities and intelligent document processing.

## 🚀 Features

- **PDF Processing**: Upload PDFs via file upload or URL
- **AI-Powered Q&A**: Answer multiple questions about PDF content
- **Vector Search**: Uses Qdrant for efficient document retrieval
- **Batch Processing**: Handles large documents with timeout-resistant batching
- **Multiple Endpoints**: Both HackRX-compliant and general-purpose APIs

## 🛠️ Technology Stack

- **FastAPI**: Modern Python web framework
- **Google Generative AI**: Gemini-1.5-flash for answer generation
- **Qdrant**: Vector database for document search
- **LangChain**: Document processing and text splitting
- **PyPDF**: PDF parsing and text extraction

## 📋 Prerequisites

- Python 3.8+
- Google Cloud API Key (for Generative AI)
- Qdrant Cloud account or local instance
- Required Python packages (see requirements.txt)

## ⚙️ Environment Setup

Create a `.env` file in the project root:

```env
QUADRANT_API_KEY=your_qdrant_api_key
QUADRANT_URI=your_qdrant_cluster_url
QUADRANT_COLLECTION=your_collection_name
GOOGLE_API_KEY=your_google_api_key
```

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd bajaj-hack
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Endpoints

### Main Endpoint (HackRX Format)

#### `POST /hackrx/run`
Process a PDF document and answer multiple questions in one request.

**Request:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six (36) months...",
        "Yes, the policy covers maternity expenses..."
    ]
}
```

### Legacy Endpoints

#### `POST /submit-pdf/`
Upload and process a PDF document.

**Form Data:**
- `file`: PDF file upload
- `pdf_url`: URL to PDF document

**Response:**
```json
{
    "message": "PDF processed",
    "filename": "generated-uuid.pdf"
}
```

#### `POST /chat/`
Ask a single question about previously uploaded documents.

**Request:**
```json
{
    "question": "What is the waiting period for cataract surgery?"
}
```

**Response:**
```json
{
    "question": "What is the waiting period for cataract surgery?",
    "answer": "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "context_snippets": ["relevant document chunks..."]
}
```

## 🔍 How It Works

### 1. Document Processing
- Downloads PDF from URL or accepts file upload
- Extracts text using PyPDFLoader
- Splits content into 300-character chunks with 50-character overlap
- Generates embeddings using Google's embedding-001 model

### 2. Vector Storage
- Creates Qdrant collection with COSINE distance
- Processes documents in batches of 5 to prevent timeouts
- Implements resilient error handling for failed batches

### 3. Question Answering
- Performs similarity search to find relevant document chunks
- Constructs context from top 5 most relevant chunks
- Uses Gemini-1.5-flash to generate answers based on context

## 🔧 Batch Processing Logic

The application uses intelligent batching to handle large documents:

```python
# Why batching? Large documents create many chunks, and uploading all at once can cause:
# 1. Network timeouts (Qdrant connection limits)
# 2. Memory issues (too many embeddings at once)
# 3. API rate limits (embedding service limitations)

batch_size = 5  # Process 5 documents per batch
# Processes chunks in batches using sliding window approach
# Continues processing even if individual batches fail
```

## 📊 Performance Considerations

- **Chunk Size**: 300 characters for optimal context relevance
- **Batch Size**: 5 documents per batch for reliability
- **Embedding Model**: `embedding-001` (768 dimensions) for speed
- **Search Results**: Top 5 relevant chunks for context
- **Error Recovery**: Failed batches don't stop processing

## 🐳 Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t hackrx-pdf-api .
docker run -p 8000:8000 --env-file .env hackrx-pdf-api
```

## 🚀 Railway Deployment

The application includes `railway.json` for easy deployment:

```json
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT"
  }
}
```

## 🧪 Testing

**Test the main endpoint:**
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://example.com/policy.pdf",
       "questions": ["What is covered under this policy?"]
     }'
```

**Health check:**
```bash
curl http://localhost:8000/docs
```

## 📈 Monitoring

The application provides detailed logging:
- Document processing progress
- Batch processing status
- Error tracking and recovery
- Vector search results
- AI response generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**1. Qdrant Connection Timeout**
- Reduce batch size in the code
- Check Qdrant cluster status
- Verify API credentials

**2. Google API Errors**
- Verify API key is valid
- Check quota limits
- Ensure billing is enabled

**3. PDF Processing Fails**
- Ensure PDF URL is accessible
- Check file format (must be PDF)
- Verify network connectivity

### Support

For issues related to:
- **Qdrant**: Check the [Qdrant documentation](https://qdrant.tech/documentation/)
- **Google AI**: Review [Google AI documentation](https://ai.google.dev/)
- **FastAPI**: See [FastAPI documentation](https://fastapi.tiangolo.com/)

---

Built with ❤️ for HackRX Competition