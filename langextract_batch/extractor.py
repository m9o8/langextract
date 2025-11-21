"""
Universal Batch Extractor for LangExtract (Vertex AI + AI Studio).

Main features / fixes:

- Supports TWO modes:
    * AI_STUDIO (api_key)
    * VERTEX_AI (project + location, via Vertex BatchPrediction)

- FILE MODE (Vertex):
    * Uses JSONL with the schema:

        {
          "instances": [
            {
              "id": "<doc_id>",
              "contents": [
                {
                  "role": "user",
                  "parts": [{ "text": "<prompt>" }]
                }
              ]
            }
          ],
          "parameters": {
            "temperature": <float>
          }
        }

    * This schema is accepted by Gemini on Vertex BatchPrediction.
    * The ID is stored in instances[0].id and is returned in outputs.

- INLINE MODE:
    * Kept mainly for AI Studio / small batches.
    * Not used in your main_vertex.py (you pass use_inline=False).

- OUTPUT PARSING:
    * For Vertex file mode, results are read from batch_job.dest.gcs_uri.
    * All *.jsonl shards under that prefix are downloaded and parsed.
    * Each line is expected to contain either:
        - "response": { "candidates": [...] }
        - or "predictions": [ { "content" / "contents": [...] } ]
    * The original document id is recovered from:
        - resp["instance"]  (some formats), or
        - resp["instances"][0]

- ALIGNMENT:
    * For Vertex/file mode, alignment is done by ID, not by position.
    * For inline mode, fallback to zip(responses, prompts).
"""

import json
import time
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional

from google import genai
from google.cloud import storage

from langextract import prompting, resolver
from langextract.core import data
from langextract.core import format_handler as fh


class BatchMode(Enum):
    AI_STUDIO = "ai_studio"
    VERTEX_AI = "vertex_ai"


class UniversalBatchExtractor:
    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        prompt_description: Optional[str] = None,
        examples: Optional[List[Any]] = None,
        format_type: data.FormatType = data.FormatType.JSON,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
    ):
        # ------------------------------------------------------------------
        # MODE SELECTION: AI Studio (api_key) vs Vertex AI (project+location)
        # ------------------------------------------------------------------
        if api_key:
            self.mode = BatchMode.AI_STUDIO
            self.client = genai.Client(api_key=api_key)
            print("✓ Using Google AI Studio")
        else:
            self.mode = BatchMode.VERTEX_AI
            self.client = genai.Client(vertexai=True, project=project, location=location)
            print("✓ Using Vertex AI BatchPrediction")

        # ------------------------------------------------------------------
        # GCS SETUP
        # ------------------------------------------------------------------
        self.gcs_bucket = None
        self.storage_client = None
        if gcs_bucket:
            self.storage_client = storage.Client(project=project)
            self.gcs_bucket = self.storage_client.bucket(gcs_bucket)
            print(f"✓ GCS bucket: gs://{gcs_bucket}")

        # ------------------------------------------------------------------
        # LangExtract PROMPTING + PARSING
        # ------------------------------------------------------------------
        self.model_id = model_id

        self.prompt_template = prompting.PromptTemplateStructured(
            description=prompt_description
        )
        self.prompt_template.examples.extend(examples or [])

        self.format_handler = fh.FormatHandler(
            format_type=format_type,
            use_wrapper=True,
            wrapper_key=data.EXTRACTIONS_KEY,
            use_fences=True,
            attribute_suffix=data.ATTRIBUTE_SUFFIX,
        )

        self.prompt_generator = prompting.QAPromptGenerator(
            template=self.prompt_template,
            format_handler=self.format_handler,
        )

        self.resolver = resolver.Resolver(format_handler=self.format_handler)

    # ======================================================================
    # PUBLIC ENTRYPOINT
    # ======================================================================
    def process_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_name: Optional[str] = None,
        temperature: float = 0.0,
        max_wait_hours: float = 24,
        poll_interval_minutes: float = 5,
        use_inline: Optional[bool] = None,
    ) -> List[data.AnnotatedDocument]:
        """
        Process a batch of documents.

        documents: list of dicts with keys:
            - "id": document id
            - "text": document full text
        """

        batch_name = batch_name or f"batch_{int(time.time())}"

        print("\n============================================================")
        print(f"Processing {len(documents)} documents")
        print(f"Mode: {self.mode.value}")
        print("============================================================")

        # 1) Generate prompts
        prompts = self._generate_prompts(documents)
        print(f"✓ Generated {len(prompts)} prompts")

        # 2) Decide INLINE vs FILE batch mode
        if use_inline is None:
            total_size = sum(len(p["prompt"]) for p in prompts)
            # Safe heuristic: inline only for AI Studio and small total size
            use_inline = (self.mode == BatchMode.AI_STUDIO and total_size < 15_000_000)

        print(f"✓ Using {'inline' if use_inline else 'file'} batch mode")

        # 3) Prepare src for batches.create
        if use_inline:
            src = self._create_inline_requests(prompts, temperature)
        else:
            src = self._create_file_requests(prompts, batch_name, temperature)

        # 4) Submit batch job
        batch_job = self.client.batches.create(model=self.model_id, src=src)
        print(f"✓ Batch job submitted: {batch_job.name}")

        if hasattr(batch_job.dest, "gcs_uri"):
            print(f"📂 Vertex output dir: {batch_job.dest.gcs_uri}")

        print("Monitor job at: https://console.cloud.google.com/vertex-ai/batch-predictions")

        # 5) Wait for job completion
        batch_job = self._wait_for_completion(
            batch_job.name,
            max_wait_seconds=max_wait_hours * 3600,
            poll_interval_seconds=poll_interval_minutes * 60,
        )

        print("✓ Batch job completed!")

        # 6) Parse results
        results = self._process_results(batch_job, prompts, use_inline=use_inline)

        print(f"✓ Parsed {len(results)} documents")
        return results

    # ======================================================================
    # PROMPT GENERATION
    # ======================================================================
    def _generate_prompts(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in documents:
            prompt = self.prompt_generator.render(question=d["text"])
            out.append({"id": d["id"], "text": d["text"], "prompt": prompt})
        return out

    # ======================================================================
    # INLINE REQUESTS (MAINLY FOR AI STUDIO)
    # ======================================================================
    def _create_inline_requests(self, prompts: List[Dict[str, Any]], temperature: float):
        """
        Inline mode: send requests directly (no JSONL, no GCS).
        Mostly relevant for AI Studio, not used by your main pipeline.
        """
        requests = []
        for p in prompts:
            requests.append(
                {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": p["prompt"]}],
                        }
                    ],
                    "parameters": {
                        "temperature": temperature,
                    },
                }
            )
        return requests

    # ======================================================================
    # FILE REQUESTS (VERTEX BATCHPREDICTION, JSONL ON GCS)
    # ======================================================================
    def _create_file_requests(
        self,
        prompts: List[Dict[str, Any]],
        batch_name: str,
        temperature: float,
    ) -> str:
        """
        Create a JSONL file for Vertex BatchPrediction with the schema:

            {
              "instances": [
                {
                  "id": "<doc_id>",
                  "contents": [
                    {
                      "role": "user",
                      "parts": [{ "text": "<prompt>" }]
                    }
                  ]
                }
              ],
              "parameters": {
                "temperature": <float>
              }
            }

        This format is accepted by Gemini on Vertex and ensures that
        the 'id' is present in the output lines.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            for p in prompts:
                entry = {
                    "instances": [
                        {
                            "id": p["id"],
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [{"text": p["prompt"]}],
                                }
                            ],
                        }
                    ],
                    "parameters": {
                        "temperature": temperature,
                    },
                }
                tmp.write(json.dumps(entry) + "\n")

            local_path = tmp.name

        # Upload JSONL to GCS (Vertex Batch mode)
        if self.gcs_bucket:
            remote_path = f"batch-inputs/{batch_name}.jsonl"
            blob = self.gcs_bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            Path(local_path).unlink(missing_ok=True)

            gcs_uri = f"gs://{self.gcs_bucket.name}/{remote_path}"
            print(f"📤 Uploaded JSONL: {gcs_uri}")
            return gcs_uri

        # Fallback: local path (if no GCS bucket configured)
        return local_path

    # ======================================================================
    # WAIT FOR BATCH JOB COMPLETION
    # ======================================================================
    def _wait_for_completion(
        self,
        job_name: str,
        max_wait_seconds: float,
        poll_interval_seconds: float,
    ):
        start = time.time()
        last = start

        while True:
            job = self.client.batches.get(name=job_name)
            state = str(job.state)

            if "SUCCEEDED" in state:
                print(f"  → Completed in {(time.time() - start) / 60:.1f} min")
                return job

            if any(e in state for e in ["FAILED", "CANCELLED", "EXPIRED"]):
                raise RuntimeError(f"Batch job failed: {state}")

            if time.time() - start > max_wait_seconds:
                raise TimeoutError("Batch job timeout exceeded")

            if time.time() - last > poll_interval_seconds:
                print(f"  → Status: {state}")
                last = time.time()

            time.sleep(poll_interval_seconds)

    # ======================================================================
    # PROCESS RESULTS
    # ======================================================================
    def _process_results(
        self,
        batch_job: Any,
        prompts: List[Dict[str, Any]],
        use_inline: bool,
    ) -> List[data.AnnotatedDocument]:

        # ---------------- INLINE MODE ----------------
        if use_inline:
            responses = list(batch_job.dest.inlined_responses)

            if not responses:
                print("⚠ WARNING: 0 inline responses parsed.")
                return []

            results: List[data.AnnotatedDocument] = []

            for resp, meta in zip(responses, prompts):
                try:
                    text = self._extract_text_from_response(resp, inline=True)

                    parsed = self.resolver.resolve(text, suppress_parse_errors=False)

                    aligned = list(
                        self.resolver.align(
                            extractions=parsed,
                            source_text=meta["text"],
                            token_offset=0,
                            char_offset=0,
                            enable_fuzzy_alignment=True,
                            fuzzy_alignment_threshold=0.75,
                            accept_match_lesser=True,
                        )
                    )

                    results.append(
                        data.AnnotatedDocument(
                            document_id=meta["id"],
                            text=meta["text"],
                            extractions=aligned,
                        )
                    )

                except Exception as e:
                    print(f"  ! Error (inline) with {meta['id']}: {e}")

            return results

        # ---------------- FILE MODE (VERTEX) ----------------
        output_uri = getattr(batch_job.dest, "gcs_uri", None)
        if not output_uri:
            raise RuntimeError("Vertex batch job returned no GCS output directory")

        print(f"🔍 Reading results from: {output_uri}")
        responses = self._download_result_file(output_uri)

        if not responses:
            print("⚠ WARNING: 0 file responses parsed.")
            return []

        # Map prompts by ID
        prompt_by_id: Dict[str, Dict[str, Any]] = {p["id"]: p for p in prompts}

        results: List[data.AnnotatedDocument] = []

        for resp in responses:
            try:
                # Retrieve instance info to get the original ID
                inst = resp.get("instance")
                if inst is None:
                    instances = resp.get("instances")
                    if isinstance(instances, list) and instances:
                        inst = instances[0]

                if not inst or "id" not in inst:
                    raise ValueError("Missing id in response")

                doc_id = inst["id"]
                meta = prompt_by_id.get(doc_id)

                if meta is None:
                    raise ValueError(f"Unknown document id in response: {doc_id}")

                # Extract model output text
                text = self._extract_text_from_response(resp, inline=False)

                parsed = self.resolver.resolve(text, suppress_parse_errors=False)

                aligned = list(
                    self.resolver.align(
                        extractions=parsed,
                        source_text=meta["text"],
                        token_offset=0,
                        char_offset=0,
                        enable_fuzzy_alignment=True,
                        fuzzy_alignment_threshold=0.75,
                        accept_match_lesser=True,
                    )
                )

                results.append(
                    data.AnnotatedDocument(
                        document_id=doc_id,
                        text=meta["text"],
                        extractions=aligned,
                    )
                )

            except Exception as e:
                print(f"  ! Error processing item: {e}")

        return results

    # ======================================================================
    # EXTRACT TEXT FROM RESPONSE (INLINE + FILE MODES)
    # ======================================================================
    def _extract_text_from_response(self, resp: Any, inline: bool) -> str:
        """
        Try multiple known schemas:

        - Vertex new batch format:
            {"response": {"candidates": [...]}, "instance": {...}}

        - Classic generative responses:
            {"candidates": [...]}

        - Predictions format:
            {"predictions": [ {"content" / "contents": [...]} ] }

        - Or:
            {"prediction": {"content" / "contents": [...] } }
        """

        # INLINE: resp is usually a typed object from the SDK
        if inline and not isinstance(resp, dict):
            # genai types
            if hasattr(resp, "response"):
                r = resp.response
                if hasattr(r, "text") and r.text:
                    return r.text
                if hasattr(r, "candidates") and r.candidates:
                    return r.candidates[0].content.parts[0].text

            if hasattr(resp, "candidates") and resp.candidates:
                return resp.candidates[0].content.parts[0].text

            raise ValueError("Unknown inline response format")

        # FILE MODE: resp is a dict parsed from JSONL

        # 1) Full "response" object
        if "response" in resp:
            r = resp["response"]
            if "candidates" in r and r["candidates"]:
                return r["candidates"][0]["content"]["parts"][0]["text"]

        # 2) Direct "candidates"
        if "candidates" in resp and resp["candidates"]:
            return resp["candidates"][0]["content"]["parts"][0]["text"]

        # 3) Predictions list
        if "predictions" in resp and resp["predictions"]:
            pred = resp["predictions"][0]
            content = pred.get("content") or pred.get("contents")
            if content:
                return content[0]["parts"][0]["text"]

        # 4) Single prediction
        if "prediction" in resp:
            pred = resp["prediction"]
            content = pred.get("content") or pred.get("contents")
            if content:
                return content[0]["parts"][0]["text"]

        raise ValueError(f"Unknown response keys: {list(resp.keys())}")

    # ======================================================================
    # DOWNLOAD ALL RESULT SHARDS FROM GCS
    # ======================================================================
    def _download_result_file(self, uri: str) -> List[Dict[str, Any]]:
        """
        Vertex writes MULTIPLE prediction shards.
        We download all *.jsonl under the given directory/prefix.
        """
        results: List[Dict[str, Any]] = []

        # GCS path
        if uri.startswith("gs://"):
            bucket_name, prefix = uri.replace("gs://", "").split("/", 1)
            bucket = self.storage_client.bucket(bucket_name)

            blobs = list(bucket.list_blobs(prefix=prefix))

            if not blobs:
                print("⚠ WARNING: No blobs found in output directory")
                return []

            for blob in blobs:
                if not blob.name.endswith(".jsonl"):
                    continue

                data = blob.download_as_text()
                for line in data.splitlines():
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except Exception as e:
                            print(f"  ! Error parsing JSONL line in {blob.name}: {e}")

            return results

        # Local path fallback
        p = Path(uri)
        if not p.exists():
            print("⚠ Output file not found:", uri)
            return []

        with p.open() as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except Exception as e:
                        print(f"  ! Error parsing JSONL line in local file: {e}")

        return results
