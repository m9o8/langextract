# Batch Processing for LangExtract

This directory contains a complete batch processing implementation for LangExtract that works with both AI Studio (API key) and Vertex AI (GCP).

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Install
pip install -e ".[batch]"

# 2. Get API key from ai.google.dev
export GEMINI_API_KEY="your-api-key"

# 3. Test installation
python test_batch_installation.py

# 4. Run example
python examples/batch_processing/ai_studio_example.py
```

**That's it!** You're now processing at 50% cost vs real-time API.

## 📁 What's Included

```
langextract/
├── langextract_batch/              # Batch processing module
│   ├── __init__.py                 # UniversalBatchExtractor
│   ├── extractor.py                # Core implementation
│   └── README.md                   # Full documentation
│
├── examples/batch_processing/      # Working examples
│   ├── ai_studio_example.py        # ⭐ Start here
│   ├── vertex_ai_example.py        # Enterprise GCP
│   └── large_corpus_example.py     # Large-scale processing
│
├── test_batch_installation.py      # Verify setup
├── BATCH_PROCESSING_QUICKSTART.md  # Getting started guide
├── BATCH_API_COMPARISON.md         # AI Studio vs Vertex AI
├── IMPLEMENTATION_SUMMARY.md       # Technical details
└── BATCH_README.md                 # This file
```

## 🎯 Choose Your Path

### Path 1: AI Studio (Simplest) ⭐ RECOMMENDED

**Best for**: Quick start, internal use, simplicity

```python
from langextract_batch import UniversalBatchExtractor
import langextract as lx
import os

extractor = UniversalBatchExtractor(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_id="gemini-2.5-flash",
    prompt_description="Your extraction instructions...",
    examples=[...]  # Your few-shot examples
)

documents = load_your_documents()  # [{'id': ..., 'text': ...}]

results = extractor.process_documents(
    documents=documents,
    batch_name="my_batch"
)
```

**Setup time**: 5 minutes
**Requirements**: Just an API key from ai.google.dev

### Path 2: Vertex AI (Enterprise)

**Best for**: GCP integration, monitoring UI, enterprise features

```python
extractor = UniversalBatchExtractor(
    project="your-gcp-project",
    location="us-central1",
    gcs_bucket="your-bucket",
    model_id="gemini-2.5-flash",
    prompt_description="...",
    examples=[...]
)
```

**Setup time**: 15-30 minutes
**Requirements**: GCP project, billing enabled

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [BATCH_PROCESSING_QUICKSTART.md](BATCH_PROCESSING_QUICKSTART.md) | 5-minute getting started |
| [langextract_batch/README.md](langextract_batch/README.md) | Full API documentation |
| [BATCH_API_COMPARISON.md](BATCH_API_COMPARISON.md) | AI Studio vs Vertex AI |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Technical details |

## 🔧 Installation

### Basic (AI Studio)

```bash
pip install -e ".[batch]"
```

This installs:

- `google-genai` (already a dependency)
- `google-cloud-storage` (for file mode)

### Verify Installation

```bash
python test_batch_installation.py
```

Expected output:

```
============================================================
BATCH PROCESSING INSTALLATION TEST
============================================================
Testing imports...
  ✓ langextract imported
  ✓ langextract_batch imported
  ✓ google-genai imported
  ✓ google-cloud-storage imported

Testing API key configuration...
  ✓ GEMINI_API_KEY is set (length: 39)

Testing UniversalBatchExtractor initialization...
  ✓ UniversalBatchExtractor initialized successfully
  ✓ Mode: ai_studio

============================================================
TEST SUMMARY
============================================================
  ✓ PASS: Module imports
  ✓ PASS: API key configuration
  ✓ PASS: Extractor initialization

Passed: 3, Failed: 0, Skipped: 0

🎉 All tests passed! You're ready to process your corpus.
```

## 📖 Examples

### 1. Simple Example (< 1K Documents)

**File**: [examples/batch_processing/ai_studio_example.py](examples/batch_processing/ai_studio_example.py)

Demonstrates:

- AI Studio authentication
- Inline processing mode
- Basic extraction task
- Result visualization

**Run it**:

```bash
export GEMINI_API_KEY="your-key"
python examples/batch_processing/ai_studio_example.py
```

### 2. Enterprise Example (Vertex AI)

**File**: [examples/batch_processing/vertex_ai_example.py](examples/batch_processing/vertex_ai_example.py)

Demonstrates:

- GCP project authentication
- File-based processing
- GCS integration
- Cloud Console monitoring

**Run it**:

```bash
export GCP_PROJECT_ID="your-project"
python examples/batch_processing/vertex_ai_example.py
```

## 🎓 Recipes

### Process Large Corpus

```python
from langextract_batch import UniversalBatchExtractor
import langextract as lx

# Setup
extractor = UniversalBatchExtractor(
    api_key=os.getenv("GEMINI_API_KEY"),
    prompt_description="Your extraction task...",
    examples=[...]  # Define your extraction schema
)

# Load all documents
all_documents = load_from_database()  # Your document corpus

# Process in batches (50K-200K per job recommended)
BATCH_SIZE = 50000

for i in range(0, len(all_documents), BATCH_SIZE):
    batch = all_documents[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE

    print(f"Processing batch {batch_num + 1}...")

    results = extractor.process_documents(
        documents=batch,
        batch_name=f"corpus_batch_{batch_num:04d}",
        use_inline=False  # Use file mode for large batches
    )

    # Save checkpoint
    lx.io.save_annotated_documents(
        results,
        f"batch_{batch_num:04d}.jsonl",
        "./results"
    )

    print(f"✓ Saved batch {batch_num + 1}")
    print(f"  Progress: {i + len(batch):,} / {len(all_documents):,}")
```

### Small Test Run

```python
# Test with 100 documents first
test_documents = all_documents[:100]

results = extractor.process_documents(
    documents=test_documents,
    batch_name="test_run",
    use_inline=True,  # Fast inline mode
    max_wait_hours=1
)

# Verify quality before scaling
print(f"Processed {len(results)} documents")
for doc in results[:3]:
    print(f"  {doc.document_id}: {len(doc.extractions)} extractions")
```

## ⚙️ Configuration

### Key Parameters

```python
results = extractor.process_documents(
    documents=documents,            # [{'id': str, 'text': str}]
    batch_name="my_batch",          # Job identifier
    temperature=0.0,                # 0.0 = deterministic
    max_wait_hours=24,              # Job timeout
    poll_interval_minutes=5,        # Status check frequency
    use_inline=None,                # Auto-detect (or force True/False)
)
```

### When to Use Inline vs File Mode

| Documents | Recommended Mode | Why |
|----------|-----------------|-----|
| < 1,000 | `use_inline=True` | Fast, no GCS needed |
| 1K-50K | `use_inline=False` | More reliable |
| > 50K | `use_inline=False` + GCS | Best performance |

## 🐛 Troubleshooting

### "No API key provided"

```bash
export GEMINI_API_KEY="your-key-from-ai.google.dev"
```

### "Package 'langextract_batch' not found"

```bash
pip install -e ".[batch]"
```

### "google-cloud-storage not found"

```bash
pip install google-cloud-storage
```

### Job takes longer than expected

- ✅ Normal! Batch jobs take 12-24 hours
- ✅ Worth the wait for 50% cost savings
- ✅ You don't need to keep script running

### Want real-time monitoring?

- AI Studio: Check status via code only
- Vertex AI: View in Cloud Console

## 📊 Comparison: Batch vs Real-Time

| Feature | Batch API | Real-Time API |
|---------|-----------|---------------|
| **Cost** | $0.0375/1M tokens | $0.075/1M tokens |
| **Savings** | 50% discount | - |
| **Latency** | 12-24 hours | Seconds |
| **Best for** | many docs | Interactive use |
| **Rate limits** | Very high | Lower |
| **Setup** | 5 minutes | Immediate |

## 🤝 Support

### Getting Help

- Check [langextract_batch/README.md](langextract_batch/README.md)
- Review [examples/batch_processing/](examples/batch_processing/)
- Run `python test_batch_installation.py`
