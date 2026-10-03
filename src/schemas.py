from pydantic import BaseModel, Field


class ReportSummary(BaseModel):
    """
    Schema for the structured summary of a student's practice report
    (Requirement 1).
    """

    research_problem: str = Field(
        description="The main research problem or background discussed in the report"
    )
    technical_route: str = Field(
        description="The technical approach/method used in the report"
    )
    conclusions: str = Field(
        description="The main results and conclusions of the report"
    )
    innovations: str = Field(
        description="The innovation/novel points described in the report"
    )


class ReviewIssue(BaseModel):
    """
    A single problem/issue identified during RAG-based comparison
    against the review standards (Requirement 2).
    """

    issue_type: str = Field(
        description="Category of the issue, e.g. 'weak_evidence', 'feasibility_concern', 'format_issue'"
    )
    description: str = Field(
        description="Detailed explanation of this issue"
    )
    related_standard: str = Field(
        description="Which standard/example this issue is based on"
    )


class ReviewResult(BaseModel):
    """
    The full result of the RAG-based review - the list of all issues found
    (Requirement 2).
    """

    issues: list[ReviewIssue] = Field(
        description="List of all problems found in the report"
    )


class LocatedAnnotation(BaseModel):
    """
    A single annotation/comment linked to a specific paragraph
    of the report (Requirement 3).
    """

    section_reference: str = Field(
        description="Which section/paragraph, e.g. 'Paragraph 3'"
    )
    original_excerpt: str = Field(
        description="The portion of that paragraph being commented on"
    )
    comment: str = Field(
        description="Comment for the mentor - what the problem is"
    )
    suggested_revision: str = Field(
        description="Suggestion for improvement"
    )


class AnnotationSet(BaseModel):
    """
    All paragraph-level annotations for the whole document (Requirement 3).
    """

    annotations: list[LocatedAnnotation] = Field(
        description="List of all located comments/annotations"
    )