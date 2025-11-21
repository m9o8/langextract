from google.cloud import storage
import json

# ============================================================
# CONFIG: QUI devi mettere la directory di output del batch job
# ============================================================

# Example:
output_dir = "gs://econai-gemini-bucket/batch-inputs/batch_1000_articles/dest"


# ============================================================
# FUNCTIONS
# ============================================================

def list_output_files(gcs_uri):
    print("\n=== LISTING FILES ===\n")
    if not gcs_uri.startswith("gs://"):
        print("❌ ERROR: Invalid GCS URI:", gcs_uri)
        return

    bucket_name, prefix = gcs_uri.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print("⚠️ No files found in:", gcs_uri)
        return

    for blob in blobs:
        print(" -", blob.name)


def inspect_output_files(gcs_uri, max_lines=5):
    print("\n=== INSPECTING FILE CONTENTS ===\n")

    if not gcs_uri.startswith("gs://"):
        print("❌ ERROR: Invalid GCS URI:", gcs_uri)
        return

    bucket_name, prefix = gcs_uri.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print("⚠️ No files found for inspection.")
        return

    for blob in blobs:
        if not blob.name.endswith(".jsonl"):
            continue

        print(f"\n📄 File: {blob.name}")

        text = blob.download_as_text().strip()

        if text == "":
            print(" ⚠️ EMPTY FILE")
            continue

        lines = text.splitlines()

        print(f" Showing first {min(max_lines, len(lines))} lines:\n")

        for line in lines[:max_lines]:
            try:
                parsed = json.loads(line)
                print(json.dumps(parsed, indent=2))
            except Exception:
                print(line)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("📥 Testing Vertex AI output inspector...\n")

    if output_dir == "<PUT_YOUR_VERTEX_OUTPUT_DIR_HERE>":
        print("❌ ERROR: You must set output_dir first!")
    else:
        list_output_files(output_dir)
        inspect_output_files(output_dir)

    print("\n✅ DONE\n")
