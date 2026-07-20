# E-Book Chatbot 📚🤖

A full-stack AI assistant that lets you upload your own PDFs and chat with them in real-time. It uses Hybrid Search (combining Vector Embeddings and Exact Keyword Matching) to guarantee high accuracy, ensuring the AI only answers based on your documents.

## Core Features

* **Multi-PDF Knowledge Base**: Upload as many books/documents as you want and chat with all of them simultaneously.
* **Hybrid Search Retrieval**: 
  * Uses **Pinecone** for semantic vector search (understanding the meaning behind your question).
  * Uses **MongoDB** for exact keyword text search (finding exact IDs, names, or rare acronyms).
  * Merges both searches mathematically using Reciprocal Rank Fusion (RRF).
* **Smart Semantic Chunking**: Uses LangChain and regex boundaries to intelligently split chunks on paragraphs and sentences, preserving context without memory overhead.
* **Fast Inference**: Uses the Groq API (Llama 3.1) for lightning-fast LLM responses.
* **Modern UI**: React and Vite frontend with a clean, responsive dark-mode aesthetic.

## Tech Stack

* **Frontend**: React.js, Vite, Vanilla CSS
* **Backend**: FastAPI (Python)
* **Databases**: MongoDB Atlas (Auth/Metadata/Keywords) & Pinecone (Vectors)
* **AI Tooling**: LangChain, Hugging Face Inference API (zero-memory cloud embeddings), pdfplumber

## Local Setup

### 1. Backend
1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file (use `.env.example` as a template):
   ```env
   GROQ_API_KEY=your_key
   PINECONE_API_KEY=your_key
   MONGODB_HOST=your_mongo_connection_string
   JWT_SECRET=your_secret
   HF_TOKEN=your_huggingface_token
   ```
4. Start the server:
   ```bash
   python main.py
   ```
   *The backend will run on `http://localhost:8002`.*

### 2. Frontend
1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install node dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the frontend folder:
   ```env
   VITE_API_URL=http://localhost:8002
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
   *The frontend will run on `http://localhost:5173`.*

## Deployment
* **Backend**: Ready to be deployed on Render. Set your Build Command to `pip install -r requirements.txt` and Start Command to `uvicorn main:app --host 0.0.0.0 --port $PORT`. The app has a built-in keep-awake cron task that automatically runs on Render.
* **Frontend**: Ready to be deployed on Vercel. Be sure to set `VITE_API_URL` to your live backend URL in the Vercel dashboard.
