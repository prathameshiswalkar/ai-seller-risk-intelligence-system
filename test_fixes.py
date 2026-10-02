import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simulate the path setup like in the pages
PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Testing AI Seller Risk Intelligence fixes...")
failed = False

try:
    import importlib
    genai_engine = importlib.import_module('src.inference.genai_engine')
    print("[OK] genai_engine imported successfully")

    # Test get_vector_store function
    if hasattr(genai_engine, 'get_vector_store'):
        print("[OK] get_vector_store function exists")
        vector_store = genai_engine.get_vector_store()
        print(f"[OK] Vector store loaded: {vector_store is not None}")
        if vector_store:
            print("[OK] Vector store ready for similarity search")
        else:
            failed = True
    else:
        print("[FAIL] get_vector_store function NOT found")
        failed = True

    # Test generate_risk_report function
    if hasattr(genai_engine, 'generate_risk_report'):
        print("[OK] generate_risk_report function exists")
        # Test with a simple prompt
        result = genai_engine.generate_risk_report("Test: Analyze a seller with $1000 revenue, 0.2 late rate, 0.1 negative rate")
        if (
            result.startswith("AI generation error:")
            or result.startswith("GROQ_API_KEY environment variable not set.")
            or "model_decommissioned" in result
        ):
            print("[FAIL] Risk report generation failed:", result[:100] + "...")
            failed = True
        else:
            print("[OK] Risk report generation working with new model")
            sample = result[:100] + "..." if len(result) > 100 else result
            safe_sample = sample.encode("ascii", errors="backslashreplace").decode("ascii")
            print("[OK] Sample output:", safe_sample)
    else:
        print("[FAIL] generate_risk_report function NOT found")
        failed = True

    print("\nAll checks completed.")

except Exception as e:
    error = str(e).encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"[FAIL] Error: {error}")
    import traceback
    traceback.print_exc()
    failed = True

if failed:
    sys.exit(1)
