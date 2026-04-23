# Lab 4 - API Calling to ChatGPT
## AI Product Listing Generator
**Lucas Barrios  |  Berlin, Germany  |  April 2026**

---

## 1. How the API Integration Works

The system automates e-commerce product listing generation by combining HuggingFace's fashion dataset with OpenAI's GPT-4o vision model. The integration follows a four-stage pipeline:

- **Dataset Loading:** 10 fashion products are pulled from the `ashraq/fashion-product-images-small` dataset on HuggingFace, returning rich metadata (name, category, color, season, usage) alongside PIL image objects.
- **Image Encoding:** PIL images are converted to base64 strings using BytesIO buffers, making them transmittable as JSON payloads to the API.
- **Prompt Engineering:** A structured prompt injects product metadata and instructs the model to return a JSON object with title, description, features, and SEO keywords.
- **API Call and Parsing:** The encoded image and prompt are sent to GPT-4o via the chat completions endpoint. The response is cleaned of markdown fences and parsed into a Python dictionary for structured storage.

Results are batch-processed with 2-second delays between requests to respect rate limits, and saved to `outputs/product_listings.json`.

---

## 2. Challenges Faced

### API Key Security
Keeping API keys out of version control was a priority from the beginning. The `.env` file was gitignored so it never touches the repository, and a separate `.env.example` file with a dummy placeholder was added so anyone cloning the repo knows what credentials they need without exposing the real ones. Simple setup, but it's the kind of thing that saves you a lot of headaches in real projects.

### JSON Parsing Inconsistency
GPT-4o occasionally wraps JSON responses in markdown code fences. A cleaning step was implemented to strip these before parsing, preventing JSON decode errors in production.

### Function Scope Errors
A few times during development, running the script threw NameError exceptions because functions were being added to the file in pieces rather than keeping the whole thing complete and in order. Quick fix each time, but it was a good reminder that when building iteratively, you need to make sure your full file stays clean and coherent at every stage — not just the part you're currently working on.

---

## 3. Quality of Generated Listings

All 3 processed products returned complete, well-structured JSON listings. The model demonstrated genuine visual understanding, for example, identifying a "contrasting denim collar" and "checkered pattern" on the navy shirt without these details being present in the metadata. Key quality observations:

- Titles were concise, descriptive, and SEO-appropriate within the 60-character limit.
- Descriptions averaged 150-180 words with persuasive, product-specific language.
- Feature lists were specific and visually grounded, not generic.
- SEO keyword sets were relevant and diverse, averaging 14-15 terms per product.

---

## 4. Potential Improvements

- **Dynamic Pricing:** Pull real pricing from a database or API rather than using a hardcoded default value.
- **Parallel Processing:** Use `asyncio` or `ThreadPoolExecutor` to process multiple products simultaneously and reduce total runtime.
- **Quality Scoring:** Implement automated checks for listing completeness, keyword density, and description length to flag low-quality outputs for review.
- **Cost Tracking:** Log token usage per request and calculate cumulative API costs to support budget management in production environments.
- **Multi-Image Support:** Send multiple angles of the same product to improve description accuracy and capture details invisible from a single view.

---

*GitHub: [github.com/Lucas-Barrios/ai-product-listing-generator](https://github.com/Lucas-Barrios/ai-product-listing-generator)*