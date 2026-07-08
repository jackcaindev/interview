from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


@dataclass(frozen=True)
class RagSource:
    key: str
    display_name: str
    collection_name: str
    path: Path
    description: str
    retry_terms: tuple[str, ...]


RAG_SOURCES: dict[str, RagSource] = {
    "safety": RagSource(
        key="safety",
        display_name="Safety Procedures",
        collection_name="manufacturing_safety_procedures",
        path=KNOWLEDGE_DIR / "safety_procedures.md",
        description="hazards, PPE, lockout/tagout, emergency response, spills, and safe work practices",
        retry_terms=("hazard control", "PPE", "lockout tagout", "emergency response", "permit"),
    ),
    "maintenance": RagSource(
        key="maintenance",
        display_name="Maintenance Manuals",
        collection_name="manufacturing_maintenance_manuals",
        path=KNOWLEDGE_DIR / "maintenance_manuals.md",
        description="equipment troubleshooting, inspections, repairs, service intervals, sensors, motors, and hydraulics",
        retry_terms=("troubleshooting", "preventive maintenance", "inspection", "repair", "equipment"),
    ),
    "quality": RagSource(
        key="quality",
        display_name="Quality Control Standards",
        collection_name="manufacturing_quality_control_standards",
        path=KNOWLEDGE_DIR / "quality_control_standards.md",
        description="tolerances, sampling, defects, calibration, holds, disposition, and acceptance criteria",
        retry_terms=("inspection", "acceptance criteria", "sampling", "nonconforming material", "calibration"),
    ),
}


SECTION_PATTERN = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)


def load_source_documents(source: RagSource) -> list[Document]:
    return split_markdown_sections(source.path.read_text(encoding="utf-8"), source)


def split_markdown_sections(markdown: str, source: RagSource) -> list[Document]:
    matches = list(SECTION_PATTERN.finditer(markdown))
    documents: list[Document] = []

    for index, match in enumerate(matches):
        section_title = match.group("title").strip()
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[content_start:content_end].strip()
        if not body:
            continue

        section_id = _section_id(section_title)
        citation = f"{source.path.name}#{section_id}"
        documents.append(
            Document(
                page_content=f"{section_title}\n\n{body}",
                metadata={
                    "source_key": source.key,
                    "source_name": source.display_name,
                    "source_file": source.path.name,
                    "section_id": section_id,
                    "section_title": section_title,
                    "citation": citation,
                },
            )
        )

    return documents


def _section_id(title: str) -> str:
    first_token = title.split(maxsplit=1)[0].strip()
    return first_token if re.match(r"^[A-Z]+-\d+$", first_token) else _slugify(title)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"
