"""Scan pages: section -> page mapping, page chunks with provenance/ACL, relevance remap."""

from docseek.core.models import Provenance, Section
from docseek.ingest.scans import page_section_id, remap_relevance, scan_chunks, section_pages


def _s(anchor: str, title: str) -> Section:
    doc = "internal-kb:kb-0023.md"
    return Section(
        section_id=f"{doc}#{anchor}",
        doc_id=doc,
        source="internal-kb",
        title=title,
        anchor=anchor,
        section_path=[title],
        text="x",
        doc_type="policy",
        acl=["eng-internal"],
        updated_at=None,
        provenance=Provenance(file="kb-0023.md", line_start=1, line_end=2),
    )


SECTIONS = [
    _s("sla", "SLA by service"),
    _s("scope", "Scope"),  # no heading match: body text mentions it before? no -> between
    _s("credits", "Service `credits`"),
    _s("contacts", "Contacts"),
    _s("appendix", "Appendix"),  # cut off by the page limit
]


def test_section_pages() -> None:
    pages = [
        "SLA by service\nPayments 99.95%\nService credits\nCredits are",
        "applied monthly.\nContacts\nOn-call: #payments",
    ]
    got = section_pages(SECTIONS, pages)
    assert got == {
        "internal-kb:kb-0023.md#sla": [1],
        "internal-kb:kb-0023.md#scope": [1],  # between two found headings
        "internal-kb:kb-0023.md#credits": [1, 2],  # spans the page break
        "internal-kb:kb-0023.md#contacts": [2],
    }  # appendix not rendered -> stays in the text corpus


def test_scan_chunks_carry_page_and_acl() -> None:
    scan = {
        "name": "internal_kb_0023",
        "doc_id": "internal-kb:kb-0023.md",
        "title": "SLA",
        "acl": ["eng-internal"],
        "doc_type": "policy",
        "status": "active",
        "version": 2,
        "supersedes": None,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    c1, c2 = scan_chunks(scan, ["page one", "  "], "tesseract")
    assert (
        c1.section_id
        == page_section_id("internal_kb_0023", 1)
        == ("scan:internal_kb_0023.pdf#page-1")
    )
    assert c1.chunk_id == "scan:internal_kb_0023.pdf#page-1::scan::0"
    assert c1.metadata["page"] == 1 and c1.metadata["modality"] == "scan"
    assert c1.metadata["acl"] == ["eng-internal"]
    assert c2.text == "(no text recognised)"


def test_remap_relevance_keeps_max_grade() -> None:
    replaced = {"a#1": ["scan:x.pdf#page-1"], "a#2": ["scan:x.pdf#page-1", "scan:x.pdf#page-2"]}
    got = remap_relevance({"a#1": 1, "a#2": 2, "b#1": 2}, replaced)
    assert got == {"scan:x.pdf#page-1": 2, "scan:x.pdf#page-2": 2, "b#1": 2}
