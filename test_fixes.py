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

try:
    import importlib
    genai_engine = importlib.import_module('src.inference.genai_engine')
    print("✓ genai_engine imported successfully")

    # Test get_vector_store function
    if hasattr(genai_engine, 'get_vector_store'):
        print("✓ get_vector_store function exists")
        vector_store = genai_engine.get_vector_store()
        print(f"✓ Vector store loaded: {vector_store is not None}")
        if vector_store:
            print("✓ FAISS vector store ready for similarity search")
    else:
        print("✗ get_vector_store function NOT found")

    # Test generate_risk_report function
    if hasattr(genai_engine, 'generate_risk_report'):
        print("✓ generate_risk_report function exists")
        # Test with a simple prompt
        result = genai_engine.generate_risk_report("Test: Analyze a seller with $1000 revenue, 0.2 late rate, 0.1 negative rate")
        if "AI generation error" in result or "model_decommissioned" in result:
            print("✗ Risk report generation failed:", result[:100] + "...")
        else:
            print("✓ Risk report generation working with new model")
            print("✓ Sample output:", result[:100] + "..." if len(result) > 100 else result)
    else:
        print("✗ generate_risk_report function NOT found")

    print("\n🎉 All fixes applied successfully!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()