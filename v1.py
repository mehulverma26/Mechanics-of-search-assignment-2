import os
import requests
import json
import math
import time
from collections import defaultdict
from flask import Flask, request, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("GOOGLE_CX")

if not API_KEY or not CX:
    raise ValueError("🚨 API Key or Search Engine ID is missing! Check your .env file.")

# Directories & Files
DATA_DIR = "image_data"
META_FILE = os.path.join(DATA_DIR, "metadata.json")

# Create directory if not exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def fetch_images_google(queries, total_images=1000, per_request=10):
    """Fetch images from Google Custom Search API. Returns None if quota is exceeded."""
    print("🔍 Fetching images from Google...")

    images_data = []
    query_index = 0

    while len(images_data) < total_images:
        query = queries[query_index % len(queries)]
        start_index = (len(images_data) % 90) + 1  # Google API max start index is 91

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "searchType": "image",
            "num": per_request,
            "start": start_index,
            "key": API_KEY,
            "cx": CX,
        }

        response = requests.get(url, params=params)

        if response.status_code == 429:
            print("⚠️ Google API quota exceeded. Switching to Wikipedia...")
            return None  # Signal to use Wikipedia

        if response.status_code != 200:
            print(f"❌ Google API Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        if "items" not in data:
            print(f"⚠️ No more images found for query: {query}")
            query_index += 1
            continue

        for item in data["items"]:
            images_data.append(
                {
                    "url": item.get("link"),
                    "title": item.get("title", ""),
                    "context": item["image"].get("contextLink", ""),
                }
            )

        print(f"✅ Fetched {len(images_data)} images so far...")
        time.sleep(1)  # Prevent API rate limiting

    return images_data


def fetch_images_wikipedia(queries, total_images=1000):
    """Fetch image metadata from Wikipedia when Google API quota is exceeded."""
    print("🌍 Fetching images from Wikipedia...")

    images_data = []
    for query in queries:
        query = query.replace(" ", "_")  # Format for Wikipedia API
        url = f"https://en.wikipedia.org/api/rest_v1/page/media-list/{query}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"❌ Wikipedia API Error for {query}: {response.status_code}")
            continue

        data = response.json()
        if "items" not in data:
            print(f"⚠️ No images found for Wikipedia query: {query}")
            continue

        for item in data["items"]:
            if "original" in item:
                image_url = item["original"]["source"]
            else:
                continue  # Skip if no image URL is found

            images_data.append(
                {
                    "url": image_url,
                    "title": item.get("title", ""),
                    "context": f"https://en.wikipedia.org/wiki/{query}",
                }
            )

            if len(images_data) >= total_images:
                break

        if len(images_data) >= total_images:
            break

    return images_data


def fetch_images(queries, total_images=1000):
    """Fetch images from Google first, switch to Wikipedia if needed."""
    images_data = fetch_images_google(queries, total_images)
    if images_data is None:
        images_data = fetch_images_wikipedia(queries, total_images)

    if images_data:
        with open(META_FILE, "w") as f:
            json.dump(images_data, f, indent=4)

    print(f"✅ Total fetched images: {len(images_data)}")
    return images_data


def preprocess(text):
    """Tokenizes and normalizes text."""
    return [word.lower() for word in text.split() if word.isalnum()]


def build_index():
    """Creates an inverted index from fetched images."""
    if not os.path.exists(META_FILE) or os.stat(META_FILE).st_size == 0:
        print("🛑 Metadata file not found. Fetching images...")
        queries = ["nature", "technology", "art", "science", "cars"]
        fetch_images(queries, total_images=1000)

    try:
        with open(META_FILE, "r") as f:
            images_data = json.load(f)
    except json.JSONDecodeError:
        print("❌ Error reading metadata.json. Regenerating...")
        images_data = fetch_images(["nature", "technology", "art", "science", "cars"])

    inverted_index = defaultdict(dict)
    doc_lengths = defaultdict(int)

    for i, image in enumerate(images_data):
        doc_id = str(i)
        text = preprocess(image["title"])
        term_freqs = defaultdict(int)

        for term in text:
            term_freqs[term] += 1

        for term, freq in term_freqs.items():
            inverted_index[term][doc_id] = freq

        doc_lengths[doc_id] = len(text)

    print(f"✅ Inverted index built with {len(inverted_index)} terms")
    return inverted_index, doc_lengths, len(images_data)


def search(query, inverted_index, doc_lengths, total_docs, model="bm25"):
    """Retrieve relevant images using BM25 ranking."""
    query_terms = preprocess(query)
    scores = defaultdict(float)

    if model == "bm25":
        k1, b = 1.5, 0.75
        avg_doc_len = sum(doc_lengths.values()) / total_docs if total_docs > 0 else 1

        for term in query_terms:
            if term in inverted_index:
                doc_freq = len(inverted_index[term])
                idf = (
                    math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
                    if doc_freq > 0
                    else 0
                )
                for doc_id, term_freq in inverted_index[term].items():
                    doc_len = doc_lengths[doc_id]
                    score = idf * (
                        (term_freq * (k1 + 1))
                        / (term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                    )
                    scores[doc_id] += score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Flask Web App
app = Flask(__name__)

# Build index
data, lengths, total = build_index()


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        results = search(query, data, lengths, total)

        with open(META_FILE, "r") as f:
            images_data = json.load(f)

        results = [
            (images_data[int(doc_id)]["url"], score)
            for doc_id, score in results[:10]
            if int(doc_id) < len(images_data)
        ]

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
