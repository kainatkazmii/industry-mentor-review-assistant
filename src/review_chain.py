import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, LLM_MODEL_NAME
from src.schemas import ReportSummary, ReviewResult
from src.vectorstore import get_relevant_chunks


review_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict academic reviewer. You are given the ORIGINAL TEXT of "
        "a student's full report (not just a summary), along with relevant "
        "reference chunks drawn from review standards and example reports.\n\n"
        "IMPORTANT: Before flagging any issue, first read the full original "
        "report text carefully. If something (such as quantitative metrics, "
        "section headings, or company name) is already present in the report, "
        "do not flag it as an issue - only flag things that are actually "
        "MISSING or WEAK in the report. If no issues are found, return an "
        "empty issues list - do not invent problems.\n\n"
        "You must respond with ONLY a valid JSON object, following this exact "
        "schema (do not use any other field names):\n"
        '{{"issues": [{{"issue_type": "string", "description": "string", '
        '"related_standard": "string"}}]}}\n\n'
        "Rules:\n"
        "- Field names must be exactly 'issue_type', 'description', and "
        "'related_standard' - no synonyms such as 'reference' or 'source'.\n"
        "- Do not wrap any word or phrase in quotation marks (' or \") "
        "inside string values.\n"
        "- Every field value must always be a plain string, never a bare/"
        "unquoted value.\n"
        "- Return ONLY the JSON object, with no extra text or explanation."
    ),
    (
        "human",
        "Original Full Report Text:\n{report_text}\n\n"
        "Standards/Examples (for comparison):\n{reference_chunks}\n\n"
        "Based on the standards above, what problems exist in this FULL "
        "report text? Respond with the JSON object only."
    ),
])


def _get_raw_json_response(inputs: dict, temperature: float) -> str:
    """
    Calls the LLM directly using Groq's NATIVE JSON mode (via model_kwargs
    response_format) - bypassing LangChain's with_structured_output()
    wrapper, which internally relies on tool-calling and was not reliable
    with this model for large inputs/outputs. Groq's native JSON mode
    guarantees syntactically valid JSON output without the added
    complexity of tool-calling.
    """
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=2048,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = review_prompt | llm
    response = chain.invoke(inputs)
    return response.content


def review_report(
    report_text: str,
    summary: ReportSummary,
    vectorstore,
    max_retries: int = 4,
) -> ReviewResult:
    """
    Compares the report's FULL TEXT against standards/examples, and
    returns the list of problems found (ReviewResult).
    """
    query_text = (
        f"{summary.research_problem} {summary.technical_route} "
        f"{summary.conclusions} {summary.innovations}"
    )
    relevant_chunks = get_relevant_chunks(query_text, vectorstore, k=5)
    reference_text = "\n\n---\n\n".join(relevant_chunks)

    inputs = {
        "report_text": report_text,
        "reference_chunks": reference_text,
    }

    temperatures = [0.0, 0.0, 0.15, 0.3]

    last_error = None
    for attempt, temp in enumerate(temperatures[:max_retries], start=1):
        try:
            raw_text = _get_raw_json_response(inputs, temp)

            # Step 1: parse the raw text into a Python dictionary
            parsed_dict = json.loads(raw_text)

            # Step 2: validate the dictionary against the Pydantic schema
            result = ReviewResult.model_validate(parsed_dict)

            return result
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} (temperature={temp}) failed: {e}")

    raise RuntimeError(
        f"Review chain failed even after {max_retries} attempts. "
        f"Last error: {last_error}"
    )