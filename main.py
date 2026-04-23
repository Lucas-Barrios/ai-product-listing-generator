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

# Function to create product listing prompt
def create_product_listing_prompt(product_name, price, category, additional_info=None):
    """Create a structured prompt for generating product listings."""
    prompt = f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{f'- Additional Info: {additional_info}' if additional_info else ''}

Please create a professional product listing that includes:

1. **Product Title** (catchy, SEO-friendly, 60 characters max)
2. **Product Description** (detailed, 150-200 words)
3. **Key Features** (bullet points, 5-7 items)
4. **SEO Keywords** (comma-separated, 10-15 keywords)

Format your response as JSON with this exact structure:
{{
    "title": "Product title here",
    "description": "Full description here",
    "features": ["Feature 1", "Feature 2"],
    "keywords": "keyword1, keyword2"
}}

Be specific about what you see in the image. Mention colors, materials, 
design elements and distinctive features. Avoid generic descriptions."""
    return prompt

# Test encoding and prompt
print("\nTesting image encoding...")
test_image = products_df.iloc[0]["image"]
encoded = encode_image(test_image)
print(f"✓ Image encoded successfully")
print(f"✓ Base64 string length: {len(encoded)} characters")

# Test prompt
test_product = products_df.iloc[0]
test_prompt = create_product_listing_prompt(
    product_name=test_product["productDisplayName"],
    price=49.99,
    category=test_product["masterCategory"],
    additional_info=f"{test_product['baseColour']}, {test_product['season']}, {test_product['usage']}"
)

print("\nPrompt preview:")
print(test_prompt[:300] + "...")