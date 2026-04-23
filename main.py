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
def generate_product_listing(product_row, price=49.99):
    """Send image and metadata to GPT-4 Vision and get product listing."""
    try:
        # Encode image
        encoded_image = encode_image(product_row["image"])
        
        # Create prompt
        prompt = create_product_listing_prompt(
            product_name=product_row["productDisplayName"],
            price=price,
            category=product_row["masterCategory"],
            additional_info=f"{product_row['baseColour']}, {product_row['season']}, {product_row['usage']}"
        )
        
        # Call GPT-4 Vision API
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        # Extract response text
        raw_response = response.choices[0].message.content
        
        # Parse JSON from response
        import json
        clean_response = raw_response.strip()
        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_response:
            clean_response = clean_response.split("```")[1].split("```")[0].strip()
        
        listing = json.loads(clean_response)
        return {"status": "success", "listing": listing}

    except Exception as e:
        return {"status": "error", "error": str(e)}

# Test API call on first product
print("\nGenerating listing for first product...")
result = generate_product_listing(products_df.iloc[0])

if result["status"] == "success":
    print("✓ Listing generated successfully")
    print("\n--- GENERATED LISTING ---")
    import json
    print(json.dumps(result["listing"], indent=2))
else:
    print(f"✗ Error: {result['error']}")


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