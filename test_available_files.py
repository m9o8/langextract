from google.cloud import storage

bucket = storage.Client().bucket("econai-gemini-bucket")

print("FILES IN BUCKET:")
for blob in bucket.list_blobs():
    print(" -", blob.name)
