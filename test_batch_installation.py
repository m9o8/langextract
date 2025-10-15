#!/usr/bin/env python3
"""
Quick test to verify batch processing installation and setup.

Run this to check if everything is configured correctly before processing
your 11M articles.
"""

import sys

from dotenv import load_dotenv

load_dotenv()


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import langextract as lx

        print("  ✓ langextract imported")
    except ImportError as e:
        print(f"  ✗ Failed to import langextract: {e}")
        return False

    try:
        from langextract_batch import BatchMode, UniversalBatchExtractor

        print("  ✓ langextract_batch imported")
    except ImportError as e:
        print(f"  ✗ Failed to import langextract_batch: {e}")
        print('  → Run: pip install -e ".[batch]"')
        return False

    try:
        from google import genai

        print("  ✓ google-genai imported")
    except ImportError as e:
        print(f"  ✗ Failed to import google-genai: {e}")
        print("  → Run: pip install google-genai")
        return False

    try:
        from google.cloud import storage

        print("  ✓ google-cloud-storage imported")
    except ImportError as e:
        print(f"  ✗ Failed to import google-cloud-storage: {e}")
        print("  → Run: pip install google-cloud-storage")
        return False

    return True


def test_api_key():
    """Test if API key is configured."""
    print("\nTesting API key configuration...")

    import os

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"  ✓ GEMINI_API_KEY is set (length: {len(api_key)})")
        return True
    else:
        print("  ✗ GEMINI_API_KEY not found")
        print("  → Get API key from: https://ai.google.dev")
        print("  → Set with: export GEMINI_API_KEY='your-key'")
        return False


def test_initialization():
    """Test that UniversalBatchExtractor can be initialized."""
    print("\nTesting UniversalBatchExtractor initialization...")

    import os

    import langextract as lx
    from langextract_batch import UniversalBatchExtractor

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  ⊘ Skipping (no API key)")
        return None

    try:
        extractor = UniversalBatchExtractor(
            api_key=api_key,
            model_id="gemini-2.5-flash",
            prompt_description="Test extraction",
            examples=[
                lx.data.ExampleData(
                    text="Test text",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="test", extraction_text="Test"
                        )
                    ],
                )
            ],
        )
        print("  ✓ UniversalBatchExtractor initialized successfully")
        print(f"  ✓ Mode: {extractor.mode.value}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        return False


def test_prompt_generation():
    """Test that prompt generation works."""
    print("\nTesting prompt generation...")

    import os

    import langextract as lx
    from langextract_batch import UniversalBatchExtractor

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  ⊘ Skipping (no API key)")
        return None

    try:
        extractor = UniversalBatchExtractor(
            api_key=api_key,
            model_id="gemini-2.5-flash",
            prompt_description="Extract entities",
            examples=[
                lx.data.ExampleData(
                    text="Apple is a company.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="company", extraction_text="Apple"
                        )
                    ],
                )
            ],
        )

        # Test prompt generation
        prompts = extractor._generate_prompts(
            [{"id": "test1", "text": "Google is a company."}]
        )

        if prompts and len(prompts) == 1:
            print("  ✓ Prompt generation works")
            print(f"  ✓ Generated prompt length: {len(prompts[0]['prompt'])} chars")
            return True
        else:
            print("  ✗ Prompt generation failed")
            return False

    except Exception as e:
        print(f"  ✗ Error during prompt generation: {e}")
        return False


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"  {status}: {test_name}")

    print("=" * 60)
    print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed! You're ready to process your corpus.")
        print("\nNext steps:")
        print("  1. Review examples/batch_processing/")
        print("  2. Customize prompt and examples for your use case")
        print("  3. Start with small test batch (100-1000 articles)")
        print("  4. Scale to full corpus (11M articles)")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix issues before proceeding.")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BATCH PROCESSING INSTALLATION TEST")
    print("=" * 60)

    results = {
        "Module imports": test_imports(),
        "API key configuration": test_api_key(),
        "Extractor initialization": test_initialization(),
        "Prompt generation": test_prompt_generation(),
    }

    success = print_summary(results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
