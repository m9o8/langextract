from google.cloud import storage

def upload_csv_to_gcs(bucket_name, source_file_path, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    print(f"Uploading {source_file_path} ...")
    blob.upload_from_filename(source_file_path)
    
    print(f"File uploaded successfully: gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    # EXAMPLE:
    upload_csv_to_gcs(
        bucket_name="econai-gemini-bucket",
        source_file_path="data/1000_articles.csv",  
        destination_blob_name="raw/articles.csv" # How it will be named on GCS
    )