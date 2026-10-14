# 🤖 Industry Mentor Review Assistant

[![Python 3.10+](https://shields.io)](https://python.org)
[![Framework-LangChain](https://shields.io)](https://langchain.com)
[![Database-ChromaDB](https://shields.io)](https://trychroma.com)

An LLM-powered, production-ready intelligent review assistant engineered to accelerate how industry mentors evaluate graduate practice reports through automated, paragraph-level semantic feedback.

---

## 📌 Problem Statement
Industry mentors supervising professional-degree graduate students are usually busy engineers or managers. They have limited bandwidth to deeply analyze every lengthy practice report, project proposal, or technical document. This resource bottleneck results in:
* **Slow review cycles** that delay student academic progression.
* **Vague, high-level feedback** (e.g., *"needs more depth"*) which lacks actionable remediation steps.

---

## 🛠️ Key Capabilities

| Feature | Technical Implementation | Core Functionality |
| :--- | :--- | :--- |
| **Structured Summarization** | `PyPDF2`, `python-docx`, `Pydantic` | Parses `.docx`/`.pdf` files to instantly extract core research problems, technical routes, conclusions, and innovations. |
| **RAG-Based Comparison** | `ChromaDB`, `HuggingFace` Embeddings | Benchmarks the document against a local vector database containing institutional evaluation standards and high-quality exemplars. |
| **Paragraph-Located Annotations** | Groq Native JSON Mode, `FastAPI` logic | Maps identified gaps directly back to the original source paragraph and auto-generates contextual revision notes. |

---

## 🎛️ Tech Stack & Tooling

* **Orchestration:** LangChain (LLM calls, prompts management, pipeline chaining)
* **Inference Engine:** Groq API (`openai/gpt-oss-20b` or Llama-based low-latency models)
* **Vector Store & Embeddings:** ChromaDB + HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Data Validation:** Pydantic (Strict schema enforcement for structured output parsing)
* **GUI Frontend:** Streamlit Dashboard

---

## 📁 Project Structure

```bash
industry_mentor_agent/
├── app.py                       # Streamlit GUI entry point
├── requirements.txt             # Project dependencies
├── src/
│   ├── config.py                # Environment variables & constants
│   ├── schemas.py               # Pydantic models for structured output
│   ├── document_loader.py       # Robust .docx/.pdf structural extraction
│   ├── summarizer.py            # Structured core components parser
│   ├── vectorstore.py           # RAG knowledge store initialization
│   ├── review_chain.py          # Vector similarity semantic search engine
│   └── annotator.py             # Paragraph-level feedback synthesis logic
├── data/
│   ├── standards/               # Evaluation criteria documents
│   ├── examples/                # High-scoring baseline reports
│   └── sample_reports/          # Unseen testing reports used in development
└── tests/
    └── test_document_loader.py   # Automated pytest unit tests
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository & Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables:**
   Create a `.env` file in the project root directory:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

3. **Build the Vector Database (One-time ingestion):**
   ```bash
   python -c "from src.vectorstore import build_vectorstore; build_vectorstore('data/standards', 'data/examples')"
   ```

4. **Launch the Application UI:**
   ```bash
   streamlit run app.py
   ```

5. **Run Automated Unit Tests:**
   ```bash
   pip install pytest
   ```
   ```bash
   pytest tests/ -v
   ```

---

## 💡 Production Architecture & Design Notes

* **Context Retention Strategy:** To mitigate loss of granular metrics, the **RAG comparison pipeline queries the raw, full-text document** against institutional criteria instead of matching against a highly-compressed 4-field abstract.
* **Deterministic Output Parsing:** While standard LangChain tool-calling can occasionally exhibit instability under long context windows, this architecture invokes **Groq's native JSON mode** backed by a **Pydantic schema validation pipeline**. A fallback retry loop with dynamic temperature scaling is implemented to gracefully handle malformed responses.

---

## 🛑 Limitations & Roadmap
* Current data parsing supports standalone `.docx` and `.pdf` documents; future iterations will support batch processing pipelines.
* The local knowledge base utilizes a lightweight illustrative set; production systems can scale horizontally to cloud vector services.
