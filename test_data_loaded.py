from google.cloud import storage

client = storage.Client()
bucket = client.bucket("econai-gemini-bucket")

print("Files in bucket:")
for blob in bucket.list_blobs():
    print(" -", blob.name)
