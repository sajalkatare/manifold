import hashlib
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SAMPLE = """Section 7.3 - Liability and Damages

The parties agree that in the event of a breach of Section 7.3, the liable party shall pay damages not exceeding $500,000. This limitation does not apply to breaches involving gross negligence or willful misconduct as defined in Section 2.4.

Notwithstanding the foregoing, neither party shall be liable for indirect, incidental, special, or consequential damages."""


def get_text():
    p = Path(__file__).resolve().parent / "test_documents" / "02_legal_contract.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return SAMPLE


def chunk_document_with_metadata(text, extraction_metadata, access_metadata):
    """
    Chunk document preserving rich metadata.

    Args:
        text: Extracted text
        extraction_metadata: From Layer 1 (source, pages, quality)
        access_metadata: For Layer 6 (department, access_level)

    Returns:
        List of LangChain Document objects with metadata
    """
    doc_hash = hashlib.md5(text.encode()).hexdigest() # sha26 can also be used
    # instead create the hash of page level/ chunk text and metadata

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=int(512 * 0.2),
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    documents = []
    for idx, chunk_text in enumerate(chunks):
        chunk_metadata = {
            **extraction_metadata,
            **access_metadata,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "chunk_length": len(chunk_text),
            "doc_hash": doc_hash, # chunk hash
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        doc = Document(page_content=chunk_text, metadata=chunk_metadata)
        documents.append(doc)

    return documents


def main():
    print("=" * 60)
    print("DEMO: Chunking with Metadata")
    print("=" * 60)

    extracted_text = get_text()

    extraction_meta = {
        "source_file": "Q4_Report.pdf",
        "doc_type": "pdf",
        "extraction_method": "pdfplumber",
        "extraction_confidence": 0.94,
    }

    access_meta = {
        "department": "finance",
        "access_level": "confidential",
        "tenant_id": "company-123",
    }

    docs = chunk_document_with_metadata(
        text=extracted_text,
        extraction_metadata=extraction_meta,
        access_metadata=access_meta,
    )

    print(f"\nCreated {len(docs)} chunks with rich metadata\n")
    for i, doc in enumerate(docs[:2], 1):
        print(f"Chunk {i} metadata:")
        for k, v in doc.metadata.items():
            print(f"  {k}: {v}")
        print(f"  page_content: {doc.page_content}")
        print(f"  page_content length: {len(doc.page_content)}")
        print()

    print("✅ Each chunk carries extraction + access metadata for filtering and audit")
    print("=" * 60)


if __name__ == "__main__":
    main()
