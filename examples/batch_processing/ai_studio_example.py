#!/usr/bin/env python3
"""
Example: Batch processing with AI Studio (Simple API key approach).

This example demonstrates the simplest way to use batch processing:
- No GCP project required
- Just an API key from ai.google.dev
- 50% cost savings vs real-time API
"""

import os

from dotenv import load_dotenv

import langextract as lx
from langextract_batch import UniversalBatchExtractor

load_dotenv()


def main():
    """Run batch extraction with AI Studio."""

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set GEMINI_API_KEY environment variable.\n"
            "Get your key at: https://ai.google.dev"
        )

    # Define your extraction task
    prompt_description = """
    Extract the following entities from news documents:
    - People (names of individuals)
    - Organizations (companies, institutions)
    - Locations (cities, countries)
    - Events (significant happenings)

    Use exact text spans from the document. Do not paraphrase.
    """

    # Provide few-shot examples
    examples = [
        lx.data.ExampleData(
            text=(
                "Apple CEO Tim Cook announced the iPhone 15 at an event in"
                " Cupertino, California."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="organization",
                    extraction_text="Apple",
                    attributes={"type": "technology company"},
                ),
                lx.data.Extraction(
                    extraction_class="person",
                    extraction_text="Tim Cook",
                    attributes={"role": "CEO"},
                ),
                lx.data.Extraction(
                    extraction_class="product",
                    extraction_text="iPhone 15",
                    attributes={"category": "smartphone"},
                ),
                lx.data.Extraction(
                    extraction_class="location",
                    extraction_text="Cupertino, California",
                    attributes={"type": "city"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text=(
                "The World Health Organization reported a breakthrough in malaria"
                " research from scientists in Geneva."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="organization",
                    extraction_text="World Health Organization",
                    attributes={"type": "international organization"},
                ),
                lx.data.Extraction(
                    extraction_class="location",
                    extraction_text="Geneva",
                    attributes={"type": "city"},
                ),
                lx.data.Extraction(
                    extraction_class="event",
                    extraction_text="breakthrough in malaria research",
                    attributes={"domain": "medical research"},
                ),
            ],
        ),
    ]

    # Initialize batch extractor
    extractor = UniversalBatchExtractor(
        api_key=api_key,
        model_id="gemini-2.0-flash-lite",
        prompt_description=prompt_description,
        examples=examples,
    )

    # Sample documents to process
    documents = [
        {
            "id": "doc_001",
            "text": (
                "Microsoft announced a partnership with OpenAI to integrate"
                " ChatGPT into Bing search. CEO Satya Nadella spoke at the"
                " company's headquarters in Redmond, Washington."
            ),
        },
        {
            "id": "doc_002",
            "text": (
                "The United Nations climate summit in Dubai concluded with a new"
                " agreement. Secretary-General António Guterres praised the"
                " participating nations for their commitment."
            ),
        },
        {
            "id": "doc_003",
            "text": (
                "Tesla's Gigafactory in Austin, Texas produced its millionth"
                " vehicle. Elon Musk celebrated the milestone with factory"
                " workers."
            ),
        },
    ]

    print("=" * 70)
    print("AI Studio Batch Processing Example")
    print("=" * 70)
    print(f"Processing {len(documents)} documents...")
    print("Cost: 50% discount vs real-time API")
    print("=" * 70)

    # Process documents
    results = extractor.process_documents(
        documents=documents,
        batch_name="ai_studio_demo",
        temperature=0.0,
        max_wait_hours=4,  # For demo, shorter timeout
        poll_interval_minutes=5,  # Check more frequently for demo
        use_inline=True,  # Use inline mode for small batch
    )

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for doc in results:
        print(f"\n📄 Document: {doc.document_id}")
        print(f"   Text: {doc.text[:100]}...")
        print(f"   Extractions: {len(doc.extractions)}")

        for extraction in doc.extractions:
            print(f"   • {extraction.extraction_class}: '{extraction.extraction_text}'")
            if extraction.attributes:
                for key, value in extraction.attributes.items():
                    print(f"     - {key}: {value}")

    # Save results
    output_file = "ai_studio_results.jsonl"
    lx.io.save_annotated_documents(results, output_name=output_file, output_dir=".")
    print(f"\n✓ Results saved to: {output_file}")

    # Generate visualization
    html_content = lx.visualize(output_file)
    viz_file = "ai_studio_results.html"
    with open(viz_file, "w") as f:
        if hasattr(html_content, "data"):
            f.write(html_content.data)
        else:
            f.write(html_content)
    print(f"✓ Visualization saved to: {viz_file}")


if __name__ == "__main__":
    main()
