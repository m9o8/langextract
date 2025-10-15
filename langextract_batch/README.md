# LangExtract Batch Processing

**50% cost savings for large-scale extraction tasks**

This module provides batch processing capabilities for LangExtract using Google's Batch APIs. Process millions of documents at half the cost of real-time inference.

## Quick Start

### AI Studio (Simplest - API Key Only)

```python
import os
from langextract_batch import UniversalBatchExtractor
import langextract as lx

# Initialize with API key
extractor = UniversalBatchExtractor(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_id="gemini-2.5-flash",
    prompt_description="Extract entities from documents...",
    examples=[
        lx.data.ExampleData(
            text="Apple announced the iPhone 15.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="company",
                    extraction_text="Apple"
                )
            ]
        )
    ]
)

# Process documents
documents = [
    {'id': 'doc1', 'text': 'Your document text...'},
    {'id': 'doc2', 'text': 'Another document...'},
]

results = extractor.process_documents(
    documents=documents,
    batch_name="my_batch"
)

# Save results
lx.io.save_annotated_documents(results, "results.jsonl", ".")
```

### Vertex AI (Enterprise - GCP Project)

```python
extractor = UniversalBatchExtractor(
    project="your-gcp-project",
    location="us-central1",
    gcs_bucket="your-bucket",  # Optional
    model_id="gemini-2.5-flash",
    prompt_description="...",
    examples=[...]
)

results = extractor.process_documents(documents, batch_name="corpus_batch")
```

## Installation

### Basic Installation (AI Studio)

```bash
# Install langextract with batch support
pip install -e ".[batch]"

# Or manually install dependencies
pip install google-genai google-cloud-storage
```

### Setup

#### For AI Studio

1. Get API key from [ai.google.dev](https://ai.google.dev)
2. Set environment variable:

   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```

#### For Vertex AI

1. Enable Vertex AI in your GCP project
2. Set up authentication:

   ```bash
   gcloud auth application-default login
   # Or use service account
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
   ```

3. (Optional) Create GCS bucket:

   ```bash
   gsutil mb -p YOUR_PROJECT gs://your-batch-bucket
   ```

## Features

### Cost Savings

- **50% discount** vs real-time API
- Additional 75% discount on cached tokens
- No rate limiting issues

### Supported Modes

| Feature | AI Studio | Vertex AI |
|---------|-----------|-----------|
| **Authentication** | API key | OAuth/Service Account |
| **Setup** | 5 minutes | 15-30 minutes |
| **GCS Required** | No (optional) | Recommended |
| **Max Batch Size** | 2GB file / 20MB inline | 1GB file |
| **Monitoring UI** | Code only | Cloud Console |

### Inline vs File-Based Processing

**Inline Mode** (AI Studio only):

- Good for < 1,000 docs
- No GCS bucket needed
- Up to 20MB of requests

**File-Based Mode** (Both):

- Required for large batches
- Uses JSONL files
- Can use GCS or local files

## Usage Patterns

### Small Batch (< 1,000 documents)

```python
results = extractor.process_documents(
    documents=documents,
    batch_name="small_batch",
    use_inline=True  # Force inline mode
)
```

### Large Batch (Thousands to Millions)

```python
# Process in chunks
BATCH_SIZE = 50000

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]

    results = extractor.process_documents(
        documents=batch,
        batch_name=f"batch_{i//BATCH_SIZE:04d}",
        use_inline=False,  # Use file mode
        max_wait_hours=24,
        poll_interval_minutes=5
    )

    # Save checkpoint
    lx.io.save_annotated_documents(
        results,
        f"batch_{i//BATCH_SIZE:04d}.jsonl",
        "./results"
    )
```

### With Checkpointing (Recommended for Large Corpora)

See [examples/batch_processing/large_corpus_example.py](../examples/batch_processing/large_corpus_example.py) for a complete implementation with:

- Progress tracking
- Error recovery
- Cost estimation
- Resume from checkpoint

## API Reference

### UniversalBatchExtractor

```python
class UniversalBatchExtractor:
    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        prompt_description: str = None,
        examples: List[ExampleData] = None,
        format_type: FormatType = FormatType.JSON,
        # AI Studio
        api_key: Optional[str] = None,
        # Vertex AI
        project: Optional[str] = None,
        location: Optional[str] = None,
        # Optional
        gcs_bucket: Optional[str] = None,
    )
```

**Parameters:**

- `model_id`: Gemini model (e.g., "gemini-2.5-flash")
- `prompt_description`: Extraction instructions
- `examples`: Few-shot examples for LangExtract
- `format_type`: JSON or YAML output
- `api_key`: AI Studio API key (simple mode)
- `project`: GCP project ID (Vertex AI)
- `location`: GCP region (Vertex AI)
- `gcs_bucket`: GCS bucket for large files (optional)

### process_documents()

```python
def process_documents(
    self,
    documents: List[Dict[str, str]],
    batch_name: Optional[str] = None,
    temperature: float = 0.0,
    max_wait_hours: int = 24,
    poll_interval_minutes: int = 5,
    use_inline: Optional[bool] = None,
) -> List[AnnotatedDocument]
```

**Parameters:**

- `documents`: List of `{'id': str, 'text': str}` dicts
- `batch_name`: Identifier for tracking
- `temperature`: Sampling temperature (0.0 = deterministic)
- `max_wait_hours`: Maximum wait time
- `poll_interval_minutes`: Status check frequency
- `use_inline`: Force inline mode (auto-detect if None)

**Returns:** List of `AnnotatedDocument` objects with extractions

## Examples

### 1. AI Studio Example

See [examples/batch_processing/ai_studio_example.py](../examples/batch_processing/ai_studio_example.py)

Simple API key-based batch processing with inline mode.

### 2. Vertex AI Example

See [examples/batch_processing/vertex_ai_example.py](../examples/batch_processing/vertex_ai_example.py)

Enterprise GCP setup with optional GCS integration.

### 3. Large Corpus Processing

See [examples/batch_processing/large_corpus_example.py](../examples/batch_processing/large_corpus_example.py)

Complete example for processing millions of documents with:

- Chunking strategy
- Checkpointing
- Cost estimation
- Progress tracking

## Troubleshooting

### "No API key provided"

```bash
export GEMINI_API_KEY="your-key"
# Get key from: https://ai.google.dev
```

### "Project not found" (Vertex AI)

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
```

### "Permission denied" (GCS)

Ensure service account has roles:

- `roles/aiplatform.user`
- `roles/storage.admin`

### Job takes longer than expected

- Normal for large batches (up to 24h)
- Check status in Cloud Console (Vertex AI)
- Or via code: `client.batches.get(name=job_name)`

### Out of memory (inline mode)

- Switch to file-based mode: `use_inline=False`
- Reduce batch size
- Use GCS bucket for large files

## Best Practices

### 1. Start Small

Test with 100-1,000 documents before processing full corpus.

### 2. Use Checkpointing

Save results after each batch for large corpora.

### 3. Monitor Costs

Run cost estimation before processing millions of documents.

### 4. Choose Right Mode

- **< 1K documents**: Inline mode (AI Studio)
- **1K-50K documents**: File mode (AI Studio or Vertex AI)
- **> 50K documents**: File mode + GCS (Vertex AI recommended)

### 5. Error Handling

Implement retry logic and save checkpoints frequently.

## Comparison: Batch vs Real-Time

| Aspect | Batch API | Real-Time API |
|--------|-----------|---------------|
| **Cost** | $0.0375/1M tokens | $0.075/1M tokens |
| **Latency** | 12-24 hours | Seconds |
| **Rate Limits** | Very high | Lower |
| **Best For** | Large corpora | Interactive use |
| **SLA** | No SLA | No SLA |

## Support

### Issues

Report issues at: [github.com/google/langextract/issues](https://github.com/google/langextract/issues)

### Documentation

- [LangExtract Main Docs](../../README.md)
- [Batch API Feasibility Analysis](../../BATCH_API_FEASIBILITY_REVISED.md)
- [AI Studio vs Vertex AI Comparison](../../BATCH_API_COMPARISON.md)

## License

Apache 2.0 - Same as LangExtract
