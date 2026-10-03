import os
import tempfile

import streamlit as st

from src.document_loader import load_document
from src.summarizer import generate_summary
from src.vectorstore import load_vectorstore
from src.review_chain import review_report
from src.annotator import generate_annotations


st.set_page_config(page_title="Industry Mentor Review Assistant", layout="wide")
st.title("📋 Industry Mentor Review Assistant")
st.write(
    "Upload a student's practice report - the system will summarize it, "
    "check it against review standards, and generate paragraph-level comments."
)

if "vectorstore" not in st.session_state:
    with st.spinner("Loading reference standards and examples..."):
        st.session_state.vectorstore = load_vectorstore()

uploaded_file = st.file_uploader(
    "Upload report (.docx or .pdf)",
    type=["docx", "pdf"],
)

if uploaded_file is not None:
    if st.button("Start Review"):

        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            with st.spinner("Parsing document..."):
                report_text = load_document(tmp_file_path)

            with st.spinner("Generating summary..."):
                summary = generate_summary(report_text)

            st.header("1️⃣ Structured Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Research Problem")
                st.write(summary.research_problem)
                st.subheader("Technical Route")
                st.write(summary.technical_route)
            with col2:
                st.subheader("Conclusions")
                st.write(summary.conclusions)
                st.subheader("Innovations")
                st.write(summary.innovations)

            with st.spinner("Checking against standards..."):
                review_result = review_report(report_text, summary, st.session_state.vectorstore)

            st.header("2️⃣ Identified Issues")
            if len(review_result.issues) == 0:
                st.success("No major issues found!")
            else:
                for issue in review_result.issues:
                    st.warning(
                        f"**{issue.issue_type}**: {issue.description}\n\n"
                        f"*Related standard: {issue.related_standard}*"
                    )

            with st.spinner("Generating paragraph-level comments..."):
                annotations = generate_annotations(report_text, review_result)

            st.header("3️⃣ Located Annotations (For Mentor Review)")
            for i, ann in enumerate(annotations.annotations, start=1):
                with st.expander(f"Annotation {i}: {ann.section_reference}"):
                    st.markdown(f"**Original text:** _{ann.original_excerpt}_")
                    st.markdown(f"**Comment:** {ann.comment}")
                    st.markdown(f"**Suggested revision:** {ann.suggested_revision}")

        finally:
            os.remove(tmp_file_path)