from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("✓ API client initialized successfully")

# Load dataset from HuggingFace
print("\nLoading product dataset...")
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train[:10]"
)
products_df = pd.DataFrame(dataset)
print(f"✓ Loaded {len(products_df)} products")

# Function to encode PIL image to base64
def encode_image(pil_image):
    """Convert PIL image to base64 string for API transmission."""
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# Test encoding on first product
print("\nTesting image encoding...")
test_image = products_df.iloc[0]["image"]
encoded = encode_image(test_image)
print(f"✓ Image encoded successfully")
print(f"✓ Base64 string length: {len(encoded)} characters")
print(f"✓ First 50 chars: {encoded[:50]}...")