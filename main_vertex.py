# file: main_process.py
import pandas as pd
from io import StringIO
from google.cloud import storage
from langextract_batch import UniversalBatchExtractor

# CONFIG
PROJECT_ID = "econai-gemini-testing"
LOCATION = "europe-west4"
BUCKET_NAME = "econai-gemini-bucket"
CSV_PATH_IN_BUCKET = "raw/articles.csv"  

def read_csv_from_gcs(bucket_name, blob_path):
    print(f"📥 Downloading CSV gs://{bucket_name}/{blob_path} ...")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    data = blob.download_as_text()
    df = pd.read_csv(StringIO(data))
    df = df.rename(columns={"an":"id", "body":"text"}) 

    print(f"✅ Loaded CSV: {df.shape[0]} rows.")
    return df

def run_pipeline():
    # 1. READ CSV FROM GCS
    df = read_csv_from_gcs(BUCKET_NAME, CSV_PATH_IN_BUCKET)

    # Ensure correct column names: 'id' and 'text'
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain columns 'id' and 'text'.")

    documents_to_process = df[['id', 'text']].to_dict(orient='records')

    # 2. INITIALIZE BATCH EXTRACTOR
    extractor = UniversalBatchExtractor(
        project="econai-gemini-testing",
        location="europe-west4",
        gcs_bucket="econai-gemini-bucket",
        prompt_description="Extract locations mentioned in the article.",
         model_id="gemini-2.5-flash")

    results = extractor.process_documents(
        documents=documents_to_process,
        batch_name="batch_1000_articles",
        use_inline=False)

    # 4. SAVE RESULTS
    print("💾 Saving results to GCS...")

    output_data = []
    for r in results:
        output_data.append({
            "id": r.document_id,
            "text": r.text,
            "extractions": r.extractions
        })

    df_out = pd.DataFrame(output_data)

    output_filename = "processed/results_1000_articles.csv"
    bucket = storage.Client().bucket(BUCKET_NAME)
    bucket.blob(output_filename).upload_from_string(df_out.to_csv(index=False))

    print(f"✅ DONE! Saved to gs://{BUCKET_NAME}/{output_filename}")

if __name__ == "__main__":
    run_pipeline()
