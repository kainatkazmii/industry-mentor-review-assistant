# Industry Mentor Review Assistant

An LLM-based intelligent review assistant that helps industry mentors review
students' practice reports faster and with more precise, paragraph-level
feedback.

## Problem

Industry mentors supervising professional-degree graduate students are
usually busy engineers/managers with limited time to deeply read every
practice report, project proposal, or technical document a student submits.
This leads to slow review cycles and vague, high-level feedback (e.g. "needs
more depth") that doesn't tell the student exactly what to fix.

## What This Tool Does

1. **Structured Summarization** — Parses a student's report (.docx/.pdf) and
   extracts a structured summary: research problem, technical route,
   conclusions, and innovation points.
2. **RAG-Based Standards Comparison** — Compares the report against a
   knowledge base of review standards and high-quality example reports
   (using retrieval-augmented generation) to identify concrete issues such as
   weak evidence, feasibility gaps, or missing context.
3. **Paragraph-Located Annotations** — Matches each identified issue to a
   specific paragraph in the original document, along with a suggested
   revision, so the mentor only needs to confirm/adjust rather than write
   feedback from scratch.

## Tech Stack

- **LangChain** — orchestration of LLM calls, prompts, and retrieval
- **Groq** (`openai/gpt-oss-20b`) — LLM inference
- **ChromaDB** + **HuggingFace sentence-transformers
- ** — vector store and
  embeddings for the RAG pipeline
- **Streamlit** — demo GUI
- **Pydantic** — structured data validation

## Project Structure

```text
industry_mentor_agent/
├── app.py                       # Streamlit GUI entry point
├── requirements.txt
├── src/
│   ├── config.py                # environment variables & constants
│   ├── schemas.py               # Pydantic schemas for structured LLM output
│   ├── document_loader.py       # .docx/.pdf text extraction
│   ├── summarizer.py            # structured summary generation
│   ├── vectorstore.py           # RAG knowledge base (standards + examples)
│   ├── review_chain.py          # RAG-based issue detection
│   └── annotator.py             # paragraph-level annotation generation
├── data/
│   ├── standards/                # review criteria documents
│   ├── examples/                 # high-quality example reports
│   └── sample_reports/           # test reports used during development
└── tests/
    └── test_document_loader.py   # basic automated tests
```



## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

   GROQ_API_KEY=your_key_here


Build the vector store (one-time step, or whenever standards/examples change):
```bash
python -c "from src.vectorstore import build_vectorstore; build_vectorstore('data/standards', 'data/examples')"
```

## Running the App

```bash
streamlit run app.py
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Design Notes

- **Summary vs. full text for review**: The RAG comparison step uses the
  full original report text (not just the compressed summary) when checking
  against standards, to avoid losing detail (e.g. exact metrics) that a
  4-field summary cannot fully capture.
- **Reliability of structured output**: Testing showed that this
  fast/free-tier model is not always fully reliable when forced into
  LangChain's tool-calling-based structured output, especially with long
  inputs. The `review_chain` and `annotator` modules call Groq's native JSON
  mode directly and validate the response against the Pydantic schema, with
  a retry loop (varying temperature) as a fallback for the rare malformed
  response.

## Limitations & Future Work

- Currently supports `.docx` and `.pdf` inputs only.
- The knowledge base (`data/standards`, `data/examples`) is a small,
  illustrative set built for this assessment; a production deployment would
  use the institution's actual program requirements and a larger archive of
  past reports.
- A larger/more capable LLM would likely reduce the need for retries in the
  structured-output steps, at higher cost.