import os
import requests
import json
import math
import cv2
import numpy as np
from collections import defaultdict
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("GOOGLE_CX")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not API_KEY or not CX or not UNSPLASH_ACCESS_KEY:
    raise ValueError("🚨 API Key or Search Engine ID is missing! Check your .env file.")

# Directories & Files
DATA_DIR = "image_data"
META_FILE = os.path.join(DATA_DIR, "metadata.json")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load OpenCV MobileNet SSD model
PROTO_FILE = "MobileNetSSD_deploy.prototxt"
MODEL_FILE = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)
CLASS_LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def annotate_image_with_opencv(image_url):
    """Enhance metadata by detecting objects in images."""
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return {"detected_objects": []}
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        detected_objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                detected_objects.append(CLASS_LABELS[class_id])
        return {"detected_objects": list(set(detected_objects))}
    except:
        return {"detected_objects": []}


### ======= Web Crawler for Image Collection ======= ###
def crawl_images(url, max_images=400):
    """Crawl images and extract metadata."""
    print(f"🕷 Crawling images from {url}...")
    images_data = []
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        images = soup.find_all("img")
        for img in images[:max_images]:
            img_url = img.get("src", "")
            alt_text = img.get("alt", "").strip()
            caption = (
                img.find_parent("figure").text.strip()
                if img.find_parent("figure")
                else ""
            )
            annotation = annotate_image_with_opencv(img_url)
            images_data.append(
                {
                    "url": img_url,
                    "alt_text": alt_text,
                    "caption": caption,
                    "detected_objects": annotation["detected_objects"],
                    "source": url,
                }
            )
    except Exception as e:
        print(f"⚠️ Error crawling {url}: {e}")
    return images_data


### ======= Fetch Images from Google ======= ###
def fetch_images_google(query, total_images=300):
    """Fetch images from Google Custom Search API."""
    print(f"📷 Fetching images from Google for '{query}'...")
    images_data = []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "cx": CX, "key": API_KEY, "searchType": "image", "num": 10}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        for item in data.get("items", []):
            annotation = annotate_image_with_opencv(item["link"])
            images_data.append(
                {
                    "url": item["link"],
                    "alt_text": str(item.get("title", "")).strip(),
                    "caption": str(item.get("snippet", "")).strip(),
                    "detected_objects": annotation["detected_objects"],
                    "source": item.get("image", {}).get("contextLink", ""),
                }
            )
            if len(images_data) >= total_images:
                break
    except Exception as e:
        print(f"⚠️ Error fetching from Google: {e}")
    return images_data


### ======= Fetch Images from Unsplash ======= ###
def fetch_images_unsplash(query, total_images=300):
    """Fetch images from Unsplash."""
    print(f"📷 Fetching images from Unsplash for '{query}'...")
    images_data = []
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query, "per_page": 10}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        for item in data.get("results", []):
            annotation = annotate_image_with_opencv(item["urls"]["regular"])
            images_data.append(
                {
                    "url": item["urls"]["regular"],
                    "alt_text": str(item.get("alt_description", "") or "").strip(),
                    "caption": str(item.get("description", "") or "").strip(),
                    "detected_objects": annotation["detected_objects"],
                    "source": item["links"]["html"],
                }
            )
            if len(images_data) >= total_images:
                break
    except Exception as e:
        print(f"⚠️ Error fetching from Unsplash: {e}")
    return images_data


### ======= Indexing (TF-IDF) ======= ###
def build_index(image_data):
    """Create an inverted index for image metadata."""
    inverted_index = defaultdict(list)
    doc_lengths = {}

    for doc_id, img in enumerate(image_data):
        # Ensure values are strings to prevent TypeErrors
        alt_text = img.get("alt_text", "") or ""
        caption = img.get("caption", "") or ""
        terms = (alt_text + " " + caption).lower().split()
        doc_lengths[doc_id] = len(terms)

        for term in set(terms):
            inverted_index[term].append(doc_id)

    return inverted_index, doc_lengths


### ======= BM25 Scoring ======= ###
def compute_bm25_scores(query, inverted_index, doc_lengths, total_docs, k1=1.5, b=0.75):
    """Compute BM25 ranking scores."""
    scores = defaultdict(float)
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1

    query_terms = query.lower().split()
    for term in query_terms:
        if term in inverted_index:
            df = len(inverted_index[term])
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_id in inverted_index[term]:
                tf = inverted_index[term].count(doc_id)
                doc_length = doc_lengths[doc_id]
                norm_tf = (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                )
                scores[doc_id] += idf * norm_tf

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


### ======= Fetch & Index Images Dynamically ======= ###
def fetch_and_index_images(query):
    """Fetch images and index dynamically."""
    images_data = (
        fetch_images_google(query, total_images=300)
        + crawl_images("https://www.nationalgeographic.com", max_images=400)
        + fetch_images_unsplash(query, total_images=300)
    )

    if images_data:
        with open(META_FILE, "w") as f:
            json.dump(images_data, f, indent=4)

    inverted_index, doc_lengths = build_index(images_data)
    total_docs = len(images_data)
    print(f"✅ {total_docs} images indexed for query: '{query}'")
    return images_data, inverted_index, doc_lengths, total_docs


### ======= Flask Web App ======= ###
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        images_data, inverted_index, doc_lengths, total_docs = fetch_and_index_images(
            query
        )
        ranked_results = compute_bm25_scores(
            query, inverted_index, doc_lengths, total_docs
        )
        results = [
            {
                "url": images_data[int(doc_id)]["url"],
                "alt_text": images_data[int(doc_id)]["alt_text"],
                "caption": images_data[int(doc_id)]["caption"],
                "detected_objects": images_data[int(doc_id)].get(
                    "detected_objects", []
                ),
                "source": images_data[int(doc_id)]["source"],
                "score": score,
            }
            for doc_id, score in ranked_results[:10]
        ]
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
