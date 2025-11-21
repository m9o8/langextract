from google.cloud import storage

BUCKET_NAME = "econai-gemini-bucket"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


blob = bucket.blob("test_upload.txt")
blob.upload_from_string("hello from python")

print("✓ Upload successful")


content = blob.download_as_text()
print("✓ Download successful:", content)


blob.delete()
print("✓ Cleaned up test file")
