import os
import requests
import json
import time
from flask import Flask, request, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("GOOGLE_CX")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

if not API_KEY or not CX or not UNSPLASH_ACCESS_KEY or not PEXELS_API_KEY:
    raise ValueError("🚨 API Key or Search Engine ID is missing! Check your .env file.")

# Directories & Files
DATA_DIR = "image_data"
META_FILE = os.path.join(DATA_DIR, "metadata.json")

# Create directory if not exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def fetch_images_pexels(queries, total_images=1000):
    """Fetch images from Pexels first."""
    print("📷 Fetching images from Pexels...")
    images_data = []
    headers = {"Authorization": PEXELS_API_KEY}

    for query in queries:
        url = "https://api.pexels.com/v1/search"
        params = {"query": query, "per_page": 30}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"❌ Pexels API Error for {query}: {response.status_code}")
            continue

        data = response.json()
        if "photos" not in data:
            print(f"⚠️ No images found for Pexels query: {query}")
            continue

        for item in data["photos"]:
            images_data.append(
                {
                    "url": item["src"]["medium"],
                    "title": item.get("alt", "Untitled"),
                    "context": item["url"],
                }
            )
            if len(images_data) >= total_images:
                break

        if len(images_data) >= total_images:
            break

    return images_data


def fetch_images_google(queries, total_images=1000, per_request=10):
    """Fetch images from Google Custom Search API if Pexels fails."""
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
            print("⚠️ Google API quota exceeded. Switching to Unsplash...")
            return None  # Signal to use Unsplash

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


def fetch_images_unsplash(queries, total_images=1000):
    """Fetch image metadata from Unsplash if Google fails."""
    print("📸 Fetching images from Unsplash...")
    images_data = []
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}

    for query in queries:
        url = "https://api.unsplash.com/search/photos"
        params = {"query": query, "per_page": 30}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"❌ Unsplash API Error for {query}: {response.status_code}")
            continue

        data = response.json()
        if "results" not in data:
            print(f"⚠️ No images found for Unsplash query: {query}")
            continue

        for item in data["results"]:
            images_data.append(
                {
                    "url": item["urls"]["regular"],
                    "title": item["alt_description"] or "Untitled",
                    "context": item["links"]["html"],
                }
            )
            if len(images_data) >= total_images:
                break

        if len(images_data) >= total_images:
            break

    return images_data


def fetch_images(queries, total_images=1000):
    """Fetch images from Pexels first, then Google, then Unsplash if needed."""
    images_data = fetch_images_pexels(queries, total_images)
    if not images_data:
        images_data = fetch_images_google(queries, total_images)
    if images_data is None or not images_data:
        images_data = fetch_images_unsplash(queries, total_images)

    if images_data:
        with open(META_FILE, "w") as f:
            json.dump(images_data, f, indent=4)

    print(f"✅ Total fetched images: {len(images_data)}")
    return images_data


# Flask Web App
app = Flask(__name__)

data = fetch_images(
    ["Marvel", "DC", "movies", "nature", "technology", "art", "science", "cars"],
    total_images=1000,
)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        results = fetch_images([query], total_images=25)
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
