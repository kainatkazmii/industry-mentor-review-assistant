from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.config import GROQ_API_KEY, LLM_MODEL_NAME
from src.schemas import ReportSummary


llm = ChatGroq(
    model=LLM_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.3,
)

summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an academic reviewer who reads students' practice reports. "
        "Extract only information that is actually present in the report. "
        "If a field is not clearly present in the report, write "
        "'Not explicitly stated in the report' - do not invent information."
    ),
    (
        "human",
        "Read the following practice report and provide a structured summary:\n\n{report_text}"
    ),
])

structured_llm = llm.with_structured_output(ReportSummary)
summary_chain = summary_prompt | structured_llm


def generate_summary(report_text: str) -> ReportSummary:
    """
    Takes the raw text of a report and generates a structured summary using the LLM.
    """
    result = summary_chain.invoke({"report_text": report_text})
    return result