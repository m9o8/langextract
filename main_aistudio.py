import os
import pandas as pd
from io import StringIO
from google.cloud import storage
from langextract_batch import UniversalBatchExtractor

# ============================================================
# CONFIGURAZIONE
# ============================================================

# Assicurati di esportare la chiave:
#   export GEMINI_API_KEY="XXXXXXXX"
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ ERROR: Environment variable GEMINI_API_KEY not set.")

BUCKET_NAME = "econai-gemini-bucket"
CSV_PATH = "raw/articles.csv"                # file da processare
OUTPUT_PATH = "processed/results_aistudio.csv"   # dove salvare l’output

MODEL = "gemini-2.5-flash"


# ============================================================
# FUNZIONI UTILI
# ============================================================

def read_csv_from_gcs(bucket: str, blob_path: str) -> pd.DataFrame:
    """Scarica e legge un CSV da Google Cloud Storage."""
    print(f"📥 Downloading CSV: gs://{bucket}/{blob_path}")

    client = storage.Client()
    blob = client.bucket(bucket).blob(blob_path)
    csv_data = blob.download_as_text()

    df = pd.read_csv(StringIO(csv_data))
    print(f"   → Loaded {df.shape[0]} rows.")

    return df


def upload_to_gcs(df: pd.DataFrame, bucket: str, dest_path: str):
    """Carica un DataFrame su GCS come CSV."""
    client = storage.Client()
    blob = client.bucket(bucket).blob(dest_path)

    blob.upload_from_string(
        df.to_csv(index=False),
        content_type="text/csv"
    )

    print(f"📤 Uploaded results to: gs://{bucket}/{dest_path}")


# ============================================================
# PIPELINE (AI STUDIO)
# ============================================================

def run_pipeline():
    print("\n============================================================")
    print("         🚀 AI STUDIO LANGEXTRACT BATCH PIPELINE")
    print("============================================================\n")

    # ------------------------------------------------------------
    # 1. Leggi CSV
    # ------------------------------------------------------------
    df = read_csv_from_gcs(BUCKET_NAME, CSV_PATH)

    if not {"an", "body"}.issubset(df.columns):
        raise ValueError("❌ CSV must contain columns: 'an' and 'body'.")

    df = df.rename(columns={"an": "id", "body": "text"})
    documents = df[["id", "text"]].to_dict(orient="records")

    print(f"🔎 Documents to process: {len(documents)}")

    # ------------------------------------------------------------
    # 2. Inizializza UniversalBatchExtractor (AI Studio mode)
    # ------------------------------------------------------------
    extractor = UniversalBatchExtractor(
        api_key=API_KEY,                # AI Studio
        model_id=MODEL,
        prompt_description="Extract locations mentioned in the article.",
        examples=[],
        gcs_bucket=None,                # PER AI STUDIO: lasciamo locale
    )

    # ------------------------------------------------------------
    # 3. Esegui batch job (file-based consigliato)
    # ------------------------------------------------------------
    print("🚀 Submitting batch job to Google AI Studio...")

    results = extractor.process_documents(
        documents=documents,
        batch_name="aistudio_batch_001",
        use_inline=False,     # AI Studio supporta JSONL file-based → più stabile
    )

    print(f"\n✅ Batch completed successfully.")
    print(f"   Parsed documents: {len(results)}")

    # ------------------------------------------------------------
    # 4. Converti risultati in DataFrame
    # ------------------------------------------------------------
    output_rows = []

    for res in results:
        output_rows.append({
            "id": res.document_id,
            "text": res.text,
            "extractions": res.extractions
        })

    df_out = pd.DataFrame(output_rows)

    # ------------------------------------------------------------
    # 5. Carica risultati su GCS
    # ------------------------------------------------------------
    upload_to_gcs(df_out, BUCKET_NAME, OUTPUT_PATH)

    print("\n🎉 DONE — AI Studio pipeline completed.\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_pipeline()