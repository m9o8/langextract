## data folder 
- *1000_articles.csv* -> articles randomly sampled from database

## test_json_key.py
Test whether the json key had been correctly created - since at the beginning we had permissions issues. 

## load_data_GC.py
Test whether the data could be correctly loaded in GC.

## langextract_batch/extractor.py
Did some changes with gpt - all changes are documented in the file itself and here a brief explanation of what i've changed. 
- I changed the code so that every jsonl request is wrapped inside a "request" object. This technically should prevent the batch job from completing with a FAILED or empty output, but this is still happening. 
- The code was reading results from "file_name", but Vertex never writes output to a GCS directory exposed through "batch_job.dest.gcs_uri", so I change this directory. 
- I changed the batch response format: instead of returning simple "predictions", now the model returns a "response" object containing "candidates"
- Vertex writes multiple jsonl shards into an output directory, but the code assumed a single output file. 

## test_available_files.py
Test that I used when having isuses with the output format. 

## delete_batch_file.py
When I changed the code to wrap every jsonl request into a "request" object, I deleted the data with the wrong format. Might be helpful to keep this to empty the whole bucket after too many mistakes. 