import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY")

def fetch_google_reviews(data_id, filename, limit=291):
    print("Fetching Google reviews.")
    all_reviews = []
    next_page_token = None

    while len(all_reviews) < limit:
        params = {
            "api_key": API_KEY,
            "engine": "google_maps_reviews",
            "data_id": data_id,
        }
        if next_page_token:
            params["next_page_token"] = next_page_token

        response = requests.get("https://serpapi.com/search", params=params).json()
        batch = response.get("reviews", [])

        if not batch:
            print("No more reviews.")
            break

        all_reviews.extend(batch)
        print(f"Collected {len(all_reviews)} Google reviews")

        next_page_token = response.get("serpapi_pagination", {}).get("next_page_token")
        if not next_page_token:
            break

    os.makedirs("data/raw", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_reviews[:limit], f, indent=4)
    print(f"Saved to {filename}")


def fetch_yelp_reviews(place_id, filename, limit=400):
    print("Fetching Yelp reviews.")
    all_reviews = []
    start = 0

    while len(all_reviews) < limit:
        params = {
            "api_key": API_KEY,
            "engine": "yelp_reviews",
            "place_id": place_id,
            "start": start,
            "num": 49,
        }

        response = requests.get("https://serpapi.com/search", params=params).json()
        batch = response.get("reviews", [])

        if not batch:
            print("No more reviews.")
            break

        all_reviews.extend(batch)
        print(f"Collected {len(all_reviews)} Yelp reviews")
        start += 49

    os.makedirs("data/raw", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_reviews[:limit], f, indent=4)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    fetch_yelp_reviews(
        place_id="auitSrx5FpFLkfHKrksYyg",
        filename="data/raw/yelp_raw.json"
    )
    fetch_google_reviews(
        data_id="0x80c2908a16708b21:0x62870c3bdfd1c32c",
        filename="data/raw/google_raw.json"
    )
