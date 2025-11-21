from google.cloud import storage

bucket_name = "econai-gemini-bucket"
blob_path = "batch-inputs/batch_1000_articles.jsonl"

client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_path)

if blob.exists():
    blob.delete()
    print(f"🗑️ Deleted: gs://{bucket_name}/{blob_path}")
else:
    print(f"❌ File not found: gs://{bucket_name}/{blob_path}")
