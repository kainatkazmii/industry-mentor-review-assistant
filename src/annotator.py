import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, LLM_MODEL_NAME
from src.schemas import ReviewResult, AnnotationSet


annotation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an academic reviewer who matches already-identified issues "
        "to numbered paragraphs. For each issue, find which paragraph number "
        "it relates to most, provide a short excerpt (original_excerpt) from "
        "that paragraph, and write a clear comment and suggested_revision. "
        "If an issue does not match a specific paragraph exactly, choose the "
        "most relevant paragraph where it should have been addressed.\n\n"
        "You must respond with ONLY a valid JSON object, following this "
        "exact schema:\n"
        '{{"annotations": [{{"section_reference": "string", '
        '"original_excerpt": "string", "comment": "string", '
        '"suggested_revision": "string"}}]}}\n\n'
        "Rules:\n"
        "- Do not wrap any word in quotation marks (' or \") inside string "
        "values - use plain text, so the JSON stays valid.\n"
        "- Every field value must always be a plain string.\n"
        "- Return ONLY the JSON object, with no extra text or explanation."
    ),
    (
        "human",
        "Numbered Paragraphs:\n{numbered_paragraphs}\n\n"
        "Identified Issues:\n{issues_text}\n\n"
        "Create one located annotation for each issue. Respond with the "
        "JSON object only."
    ),
])


def split_into_paragraphs(text: str) -> list[str]:
    """
    Splits the raw document text into a list of individual paragraphs,
    removing blank lines.
    """
    raw_parts = text.split("\n")
    paragraphs = [p.strip() for p in raw_parts if p.strip()]
    return paragraphs


def format_numbered_paragraphs(paragraphs: list[str]) -> str:
    """
    Converts a list of paragraphs into a numbered string, e.g.:
    [Paragraph 1] This is the first paragraph...
    [Paragraph 2] This is the second paragraph...
    """
    lines = []
    for i, para in enumerate(paragraphs, start=1):
        lines.append(f"[Paragraph {i}] {para}")
    return "\n\n".join(lines)


def _get_raw_json_response(inputs: dict, temperature: float) -> str:
    """
    Calls the LLM directly using Groq's native JSON mode, for the same
    reliability reasons explained in review_chain.py.
    """
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=2048,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = annotation_prompt | llm
    response = chain.invoke(inputs)
    return response.content


def generate_annotations(
    report_text: str,
    review_result: ReviewResult,
    max_retries: int = 4,
) -> AnnotationSet:
    """
    Matches identified issues to specific paragraphs of the document,
    producing located annotations.
    """
    paragraphs = split_into_paragraphs(report_text)
    numbered_paragraphs = format_numbered_paragraphs(paragraphs)

    issues_lines = []
    for issue in review_result.issues:
        issues_lines.append(
            f"- [{issue.issue_type}] {issue.description} "
            f"(Standard: {issue.related_standard})"
        )
    issues_text = "\n".join(issues_lines)

    inputs = {
        "numbered_paragraphs": numbered_paragraphs,
        "issues_text": issues_text,
    }

    temperatures = [0.0, 0.0, 0.15, 0.3]

    last_error = None
    for attempt, temp in enumerate(temperatures[:max_retries], start=1):
        try:
            raw_text = _get_raw_json_response(inputs, temp)
            parsed_dict = json.loads(raw_text)
            result = AnnotationSet.model_validate(parsed_dict)
            return result
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} (temperature={temp}) failed: {e}")

    raise RuntimeError(
        f"Annotation chain failed even after {max_retries} attempts. "
        f"Last error: {last_error}"
    )