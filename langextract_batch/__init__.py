"""
LangExtract Batch Processing Module.

Provides batch processing capabilities for LangExtract using Google's Batch APIs.
Supports both AI Studio (API key) and Vertex AI (GCP project) backends.

Usage:
    from langextract_batch import UniversalBatchExtractor

    # AI Studio (simplest)
    extractor = UniversalBatchExtractor(
        api_key="your-key",
        prompt_description="Extract entities...",
        examples=[...]
    )

    # Vertex AI (enterprise)
    extractor = UniversalBatchExtractor(
        project="your-project",
        location="us-central1",
        gcs_bucket="your-bucket",
        prompt_description="...",
        examples=[...]
    )

    # Process documents
    results = extractor.process_documents(
        documents=[{'id': '...', 'text': '...'}],
        batch_name="my_batch"
    )
"""

from langextract_batch.extractor import UniversalBatchExtractor, BatchMode

__version__ = "0.1.0"
__all__ = ["UniversalBatchExtractor", "BatchMode"]
