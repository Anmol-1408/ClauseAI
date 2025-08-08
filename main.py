from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List

from typing import Optional
from fastapi.responses import JSONResponse
import aiohttp
import os
import uuid

from decouple import config

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from langchain_google_genai import GoogleGenerativeAIEmbeddings


QUADRANT_API_KEY = config("QUADRANT_API_KEY")
QUADRANT_URI = config("QUADRANT_URI")
QUADRANT_COLLECTION = config("QUADRANT_COLLECTION")
GOOGLE_API_KEY = config("GOOGLE_API_KEY")

import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07") # this model is larger than 1000, so it is slower (1024 dimensions)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
    ) # lenght is smaller than 1000, so it is faster (768 dimensions)



# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

qdrant_client = QdrantClient(
    url = QUADRANT_URI,
    api_key = QUADRANT_API_KEY,
)

print(qdrant_client.get_collections())

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class HackRXRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]  # List of questions to answer

class HackRXResponse(BaseModel):
    answers: List[str]  # List of answers corresponding to the questions



@app.post("/hackrx/run")
async def hackrx_run(request: HackRXRequest):
    """
    Main endpoint that processes a PDF document and answers multiple questions
    Format matches the HackRX competition requirements
    """
    try:
        # Step 1: Download PDF from the provided URL
        pdf_url = request.documents
        if not pdf_url.lower().endswith(".pdf") and "pdf" not in pdf_url.lower():
            raise HTTPException(status_code=400, detail="URL does not point to a PDF")
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail="Failed to fetch PDF from URL")
                    contents = await resp.read()
            
            filename = f"{uuid.uuid4()}.pdf"
            file_path = os.path.join(UPLOAD_DIR, filename)

            with open(file_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

        # Step 2: Load and process the PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        if not pages:
            raise HTTPException(status_code=400, detail="No pages found in the PDF")
        
        print(f"[INFO] Loaded {len(pages)} pages from the PDF.")

        # Step 3: Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        all_chunks = []
        for page in pages:
            chunks = text_splitter.split_documents([page])
            all_chunks.extend(chunks)

        print(f"[INFO] Total chunks to process: {len(all_chunks)}")
        
        # Step 4: Prepare Qdrant collection
        COLLECTION_NAME = QUADRANT_COLLECTION
        
        # Check if collection exists before creating
        existing_collections = [c.name for c in qdrant_client.get_collections().collections]
        if COLLECTION_NAME not in existing_collections:
            embedding_dim = len(embeddings.embed_query("test"))
            print(f"[INFO] Embedding dimension: {embedding_dim}")
            
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )

        vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embedding=embeddings,
                )
        
        # Step 5: Store chunks in batches
        batch_size = 10
        successful_batches = 0
        failed_batches = 0
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(all_chunks) + batch_size - 1)//batch_size
            
            try:
                vector_store.add_documents(batch)
                successful_batches += 1
                if batch_num % 10 == 0:
                    print(f"[INFO] Successfully processed batch {batch_num}/{total_batches}")
            except Exception as e:
                failed_batches += 1
                print(f"[ERROR] Failed to add batch {batch_num}/{total_batches}: {str(e)}")
                continue
        
        print(f"[INFO] Processing complete: {successful_batches} successful, {failed_batches} failed batches")

        # Step 6: Answer all questions
        answers = []
        
        for question in request.questions:
            try:
                # Query the vector store for each question
                results = vector_store.similarity_search(question, k=5)
                if not results:
                    answers.append("No relevant information found in the document.")
                    continue
                
                context = "\n".join([doc.page_content for doc in results])

                prompt = f"""
You are a helpful assistant. Use the following context to answer the user's question.
Provide a clear, concise answer based only on the information provided in the context.

Context:
{context}

Question:
{question}

Answer:"""

                # Generate answer using Google Generative AI
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                answer = response.text.strip()
                answers.append(answer)
                
            except Exception as e:
                print(f"[ERROR] Failed to answer question '{question}': {str(e)}")
                answers.append(f"Error processing question: {str(e)}")

        # Clean up the uploaded file
        try:
            os.remove(file_path)
        except:
            pass

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/submit-pdf/")
async def submit_pdf(
    file: Optional[UploadFile] = File(None),
    pdf_url: Optional[str] = Form(None)
):
    # Step 1: Get PDF from either file or URL
    if file:
        file_ext = os.path.splitext(file.filename)[-1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

    # If user provided a URL
    elif pdf_url:
        if not pdf_url.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="URL does not point to a PDF")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail="Failed to fetch PDF from URL")
                    contents = await resp.read()
            
            filename = f"{uuid.uuid4()}.pdf"
            file_path = os.path.join(UPLOAD_DIR, filename)

            with open(file_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}") 

    else:
        raise HTTPException(status_code=400, detail="Please provide a PDF file or a URL")
 
    # Step 2: Load the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    if not pages:
        raise HTTPException(status_code=400, detail="No pages found in the PDF")
    print(f"[INFO] Loaded {len(pages)} pages from the PDF.")
    print(f"[INFO] Metadata of first page:\n{pages[0].metadata}")
    print(f"[INFO] Content of first page:\n{pages[0].page_content}")

    # Step 3: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    all_chunks = []
    for page in pages:
        chunks = text_splitter.split_documents([page])  # Wrap page in a list
        print(f"[INFO] Number of chunks from page: {len(chunks)}")
        if chunks:
            print(f"[INFO] First chunk content:\n{chunks[0].page_content}")
        all_chunks.extend(chunks)

    print(f"[INFO] Total chunks to process: {len(all_chunks)}")
    
    # Step 4: Prepare Qdrant collection
    COLLECTION_NAME = QUADRANT_COLLECTION
    
    # Check if collection exists before creating
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        # Get embedding dimension
        embedding_dim = len(embeddings.embed_query("test"))
        print(f"[INFO] Embedding dimension: {embedding_dim}")
        
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings,
            )
    
    # Store the chunks as vectors in batches to avoid timeout
    # Why batching? Large documents create many chunks, and uploading all at once can cause:
    # 1. Network timeouts (Qdrant connection limits)
    # 2. Memory issues (too many embeddings at once)
    # 3. API rate limits (embedding service limitations)
    
    batch_size = 5  # Process 5 documents per batch - smaller batches = more reliable but slower
    successful_batches = 0  # Track how many batches uploaded successfully
    failed_batches = 0      # Track how many batches failed (for monitoring)
    
    # Process chunks in batches using a sliding window approach
    for i in range(0, len(all_chunks), batch_size):
        # Create a batch by slicing the chunks array
        # e.g., if batch_size=5: batch 1 = chunks[0:5], batch 2 = chunks[5:10], etc.
        batch = all_chunks[i:i + batch_size]
        
        # Calculate batch number for logging (1-indexed for human readability)
        batch_num = i//batch_size + 1
        
        # Calculate total number of batches needed
        # Using ceiling division: (total_chunks + batch_size - 1) // batch_size
        total_batches = (len(all_chunks) + batch_size - 1)//batch_size
        
        try:
            # Attempt to upload this batch to Qdrant
            # This involves: embedding generation + vector storage
            vector_store.add_documents(batch)
            successful_batches += 1
            
            # Log progress every 10 batches to avoid console spam
            # Only show milestone updates to keep output clean
            if batch_num % 10 == 0:
                print(f"[INFO] Successfully processed batch {batch_num}/{total_batches}")
                
        except Exception as e:
            # If this batch fails (timeout, network error, etc.), don't crash the whole process
            # Just log the error and continue with the next batch
            failed_batches += 1
            print(f"[ERROR] Failed to add batch {batch_num}/{total_batches}: {str(e)}")
            # The 'continue' ensures we move to the next batch instead of stopping
            continue
    
    # Final summary: show overall success/failure rate
    print(f"[INFO] Processing complete: {successful_batches} successful, {failed_batches} failed batches")

    return JSONResponse({"message": "PDF processed", "filename": filename})


@app.post("/chat/")
async def chat(request: ChatRequest):
    question = request.question
    if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)
    COLLECTION_NAME = QUADRANT_COLLECTION
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        )
    
    # Step 5: Query the vector store
    results = vector_store.similarity_search(question, k=5)
    if not results:
        return JSONResponse({"error": "No relevant documents found"}, status_code=404)
    print(f"[INFO] Found {len(results)} relevant documents.")
    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{question}
        """

    # Step 6: Call Google Generative AI to get the answer
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        return JSONResponse({"error": f"Failed to generate answer: {str(e)}"}, status_code=500)
    return JSONResponse({
            "question": question,
            "answer": answer,
            "context_snippets": [doc.page_content for doc in results]
        }, status_code=200)