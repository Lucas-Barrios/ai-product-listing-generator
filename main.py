from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("✓ API client initialized successfully")

# Load dataset from HuggingFace
print("\nLoading product dataset...")
try:
    dataset = load_dataset(
        "ashraq/fashion-product-images-small", 
        split="train[:10]"  # First 10 products for testing
    )
    products_df = pd.DataFrame(dataset)
    print(f"✓ Loaded {len(products_df)} products")
    print(f"✓ Columns: {products_df.columns.tolist()}")
    print("\nSample product:")
    print(products_df.iloc[0])

except Exception as e:
    print(f"✗ Error loading dataset: {e}")