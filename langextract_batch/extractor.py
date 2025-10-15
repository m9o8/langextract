"""
Universal Batch Extractor for LangExtract.

Supports both AI Studio and Vertex AI batch processing backends.
"""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.cloud import storage

from langextract import prompting, resolver
from langextract.core import data
from langextract.core import format_handler as fh


class BatchMode(Enum):
    """Batch API backend selection."""

    AI_STUDIO = "ai_studio"  # Use API key, simpler
    VERTEX_AI = "vertex_ai"  # Use GCP project, more features


class UniversalBatchExtractor:
    """Batch extractor compatible with both AI Studio and Vertex AI.

    This class provides a unified interface for batch processing using either:
    - AI Studio API (simple, API key-based)
    - Vertex AI API (enterprise, GCP project-based)

    Both APIs provide 50% cost savings compared to real-time inference.

    Example:
        # AI Studio (simplest approach)
        extractor = UniversalBatchExtractor(
            api_key="your-api-key",
            prompt_description="Extract entities...",
            examples=[...]
        )

        # Vertex AI (enterprise approach)
        extractor = UniversalBatchExtractor(
            project="your-gcp-project",
            location="us-central1",
            gcs_bucket="your-bucket",
            prompt_description="...",
            examples=[...]
        )

        # Process documents
        results = extractor.process_documents(
            documents=[{'id': 'doc1', 'text': 'Document text...'}],
            batch_name="test_batch"
        )
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        prompt_description: str = None,
        examples: Optional[List[Any]] = None,
        format_type: data.FormatType = data.FormatType.JSON,
        # AI Studio params (simpler)
        api_key: Optional[str] = None,
        # Vertex AI params (optional, for advanced users)
        project: Optional[str] = None,
        location: Optional[str] = None,
        # Storage (optional for file-based mode)
        gcs_bucket: Optional[str] = None,
    ):
        """Initialize batch extractor.

        Args:
            model_id: Gemini model to use (e.g., "gemini-2.5-flash")
            prompt_description: Extraction instructions for the model
            examples: Few-shot examples for LangExtract
            format_type: Output format (JSON or YAML)
            api_key: API key for AI Studio (simplest option)
            project: GCP project ID (for Vertex AI)
            location: GCP region (for Vertex AI, e.g., "us-central1")
            gcs_bucket: GCS bucket name for file-based processing (optional)

        Raises:
            ValueError: If neither api_key nor (project + location) are provided
        """
        self.model_id = model_id

        # Determine mode and initialize client
        if api_key:
            self.mode = BatchMode.AI_STUDIO
            self.client = genai.Client(api_key=api_key)
            print("✓ Using AI Studio Batch API (API key authentication)")
        elif project and location:
            self.mode = BatchMode.VERTEX_AI
            self.client = genai.Client(
                vertexai=True, project=project, location=location
            )
            self.project = project
            self.location = location
            print("✓ Using Vertex AI Batch API (OAuth authentication)")
        else:
            raise ValueError(
                "Must provide either:\n"
                "  - api_key (for AI Studio), or\n"
                "  - project + location (for Vertex AI)"
            )

        # Optional GCS setup
        self.gcs_bucket = None
        if gcs_bucket:
            storage_project = project if self.mode == BatchMode.VERTEX_AI else None
            self.storage_client = storage.Client(project=storage_project)
            self.gcs_bucket = self.storage_client.bucket(gcs_bucket)
            print(f"✓ Using GCS bucket: gs://{gcs_bucket}")

        # Setup LangExtract components (no provider modification needed!)
        self.prompt_template = prompting.PromptTemplateStructured(
            description=prompt_description
        )
        self.prompt_template.examples.extend(examples or [])

        self.format_handler = fh.FormatHandler(
            format_type=format_type,
            use_wrapper=True,
            wrapper_key=data.EXTRACTIONS_KEY,
            use_fences=True,  # Model outputs fenced JSON
            attribute_suffix=data.ATTRIBUTE_SUFFIX,
        )

        self.prompt_generator = prompting.QAPromptGenerator(
            template=self.prompt_template,
            format_handler=self.format_handler,
        )

        self.resolver = resolver.Resolver(format_handler=self.format_handler)

    def process_documents(
        self,
        documents: List[Dict[str, str]],
        batch_name: Optional[str] = None,
        temperature: float = 0.0,
        max_wait_hours: int = 24,
        poll_interval_minutes: int = 5,
        use_inline: Optional[bool] = None,
    ) -> List[data.AnnotatedDocument]:
        """Process documents using batch API.

        Args:
            documents: List of dicts with 'id' and 'text' keys
            batch_name: Optional batch identifier for tracking
            temperature: Sampling temperature (0.0 = deterministic)
            max_wait_hours: Maximum hours to wait for job completion
            poll_interval_minutes: How often to check job status
            use_inline: Force inline mode (auto-detect if None)
                - AI Studio inline: good for < 1000 documents
                - File mode: better for larger batches

        Returns:
            List of AnnotatedDocuments with extractions

        Raises:
            RuntimeError: If batch job fails
            TimeoutError: If job exceeds max_wait_hours
        """
        batch_name = batch_name or f"batch_{int(time.time())}"

        print(f"\n{'=' * 60}")
        print(f"Processing {len(documents)} documents")
        print(f"Mode: {self.mode.value}")
        print("Cost: 50% discount vs real-time API")
        print(f"{'=' * 60}\n")

        # Step 1: Generate prompts
        print("Step 1/5: Generating prompts...")
        prompts_with_metadata = self._generate_prompts(documents)
        print(f"  ✓ Generated {len(prompts_with_metadata)} prompts")

        # Step 2: Decide inline vs file-based
        if use_inline is None:
            # Auto-detect based on size
            total_size = sum(len(p["prompt"]) for p in prompts_with_metadata)
            use_inline = (
                total_size < 15_000_000  # 15MB threshold (leave margin for 20MB limit)
                and self.mode == BatchMode.AI_STUDIO
            )

        print(
            f"\nStep 2/5: Preparing batch job ({'inline' if use_inline else 'file-based'})..."
        )

        # Step 3: Create requests
        if use_inline:
            requests = self._create_inline_requests(prompts_with_metadata, temperature)
            print(f"  ✓ Created {len(requests)} inline requests")
            batch_job = self.client.batches.create(model=self.model_id, src=requests)
        else:
            file_uri = self._create_file_requests(
                prompts_with_metadata, batch_name, temperature
            )
            print(f"  ✓ Uploaded batch file: {file_uri}")
            batch_job = self.client.batches.create(model=self.model_id, src=file_uri)

        # Step 4: Submit job
        print(f"\nStep 3/5: Job submitted: {batch_job.name}")
        if self.mode == BatchMode.VERTEX_AI:
            print(
                "  Monitor at:"
                " https://console.cloud.google.com/vertex-ai/batch-predictions"
            )

        # Step 5: Wait for completion
        print(
            f"\nStep 4/5: Waiting for completion of {batch_job.name}: (max {max_wait_hours}h, polling every"
            f" {poll_interval_minutes}m)..."
        )
        batch_job = self._wait_for_completion(
            batch_job.name, max_wait_hours * 3600, poll_interval_minutes * 60
        )
        print("  ✓ Job completed!")

        # Step 6: Process results
        print("\nStep 5/5: Processing results...")
        results = self._process_results(batch_job, prompts_with_metadata, use_inline)
        print(f"  ✓ Processed {len(results)} documents")

        return results

    def _generate_prompts(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Generate prompts for all documents using LangExtract."""
        prompts_with_metadata = []
        for doc in documents:
            prompt = self.prompt_generator.render(
                question=doc["text"], additional_context=None
            )
            prompts_with_metadata.append(
                {"id": doc["id"], "text": doc["text"], "prompt": prompt}
            )
        return prompts_with_metadata

    def _create_inline_requests(
        self, prompts_with_metadata: List[Dict], temperature: float
    ) -> List[Dict]:
        """Create inline request objects for batch API."""
        requests = []
        for item in prompts_with_metadata:
            requests.append(
                {
                    "contents": [{"role": "user", "parts": [{"text": item["prompt"]}]}],
                    "config": {
                        "temperature": temperature,
                    },
                }
            )
        return requests

    def _create_file_requests(
        self, prompts_with_metadata: List[Dict], batch_name: str, temperature: float
    ) -> str:
        """Create JSONL file and upload to GCS or save locally."""
        import tempfile

        # Create JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in prompts_with_metadata:
                request = {
                    "contents": [{"role": "user", "parts": [{"text": item["prompt"]}]}],
                    "config": {
                        "temperature": temperature,
                    },
                }
                f.write(json.dumps(request) + "\n")
            temp_path = f.name

        # Upload to GCS if bucket provided, otherwise use local file
        if self.gcs_bucket:
            blob_name = f"batch-inputs/{batch_name}.jsonl"
            blob = self.gcs_bucket.blob(blob_name)
            blob.upload_from_filename(temp_path)
            file_uri = f"gs://{self.gcs_bucket.name}/{blob_name}"

            # Also save metadata for result matching
            metadata_blob = self.gcs_bucket.blob(f"metadata/{batch_name}.json")
            metadata_blob.upload_from_string(
                json.dumps(
                    [
                        {"id": item["id"], "text": item["text"]}
                        for item in prompts_with_metadata
                    ]
                )
            )

            # Clean up temp file
            Path(temp_path).unlink()
        else:
            # Use local file (AI Studio supports this!)
            file_uri = temp_path

        return file_uri

    def _wait_for_completion(
        self, job_name: str, max_wait_seconds: int, poll_interval_seconds: int
    ) -> Any:
        """Poll job until completion or timeout."""
        start_time = time.time()
        last_update = start_time

        while True:
            batch_job = self.client.batches.get(name=job_name)

            # Convert state to string for comparison (handles both enum and string)
            state_str = str(batch_job.state)

            # Check completion
            if "SUCCEEDED" in state_str:
                elapsed = (time.time() - start_time) / 60
                print(f"    → Completed in {elapsed:.1f} minutes")
                return batch_job

            if any(
                status in state_str for status in ["FAILED", "CANCELLED", "EXPIRED"]
            ):
                raise RuntimeError(f"Batch job {batch_job.state}: {batch_job}")

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                raise TimeoutError(
                    f"Job exceeded max wait time of {max_wait_seconds / 3600:.1f} hours"
                )

            # Progress update every n polling seconds
            if time.time() - last_update > poll_interval_seconds:
                print(
                    f"    → Status: {batch_job.state} ({elapsed / 60:.1f} min elapsed)"
                )
                last_update = time.time()

            time.sleep(poll_interval_seconds)

    def _process_results(
        self,
        batch_job: Any,
        prompts_with_metadata: List[Dict],
        inline: bool,
    ) -> List[data.AnnotatedDocument]:
        """Extract and process results through LangExtract resolver."""
        annotated_documents = []

        # Get responses
        if inline:
            responses = batch_job.dest.inlined_responses
        else:
            responses = self._download_result_file(batch_job.dest.file_name)

        # Process each response
        for response, metadata in zip(responses, prompts_with_metadata):
            try:
                # Extract text from response (handle both object and dict formats)
                if inline:
                    # Inline response object has .response.text or .response.candidates
                    if hasattr(response, "response"):
                        if hasattr(response.response, "text"):
                            llm_output = response.response.text
                        else:
                            llm_output = (
                                response.response.candidates[0].content.parts[0].text
                            )
                    else:
                        llm_output = response.candidates[0].content.parts[0].text
                else:
                    # File-based dict format
                    llm_output = response["candidates"][0]["content"]["parts"][0][
                        "text"
                    ]

                # Clean up the output: remove "A: " prefix if present (from Q&A format)
                if llm_output.startswith("A: "):
                    llm_output = llm_output[3:].lstrip()

                # Resolve with LangExtract: two-step process
                # Step 1: Parse the model output into extractions
                parsed_extractions = self.resolver.resolve(
                    input_text=llm_output,
                    suppress_parse_errors=False,
                )

                # Step 2: Align extractions with source text
                extractions = list(
                    self.resolver.align(
                        extractions=parsed_extractions,
                        source_text=metadata["text"],
                        token_offset=0,
                        char_offset=0,
                        enable_fuzzy_alignment=True,
                        fuzzy_alignment_threshold=0.75,
                        accept_match_lesser=True,
                    )
                )

                # Create annotated document
                doc = data.AnnotatedDocument(
                    document_id=metadata["id"],
                    text=metadata["text"],
                    extractions=extractions,
                )
                annotated_documents.append(doc)

            except Exception as e:
                print(f"  ! Error processing document {metadata['id']}: {e}")
                continue

        return annotated_documents

    def _download_result_file(self, output_uri: str) -> List[Dict]:
        """Download result file from GCS or local filesystem."""
        if output_uri.startswith("gs://"):
            # Parse GCS URI
            parts = output_uri.replace("gs://", "").split("/", 1)
            bucket_name, blob_path = parts[0], parts[1]

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_text()
        else:
            # Local file
            with open(output_uri) as f:
                content = f.read()

        # Parse JSONL
        results = []
        for line in content.strip().split("\n"):
            if line:
                results.append(json.loads(line))
        return results
