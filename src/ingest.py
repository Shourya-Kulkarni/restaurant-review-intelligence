import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY")

#Fetching google reviews for a restaurant using SerpApi
def fetch_google_reviews(data_id, filename, limit=291):
    #Using a list for all accumulated reviews and fetching the next page of results
    print("Fetching Google reviews.")
    all_reviews = []
    next_page_token = None

    #Building the query paramaters for the Google Reviews engine
    while len(all_reviews) < limit:
        params = {
            "api_key": API_KEY,
            "engine": "google_maps_reviews",
            "data_id": data_id,
        }
        # Adds a pagination token from the previous request if it is there
        # Returns the next page of reviews
        if next_page_token:
            params["next_page_token"] = next_page_token

        # Makes the HTTP GET request to SerpApi and parse the JSON results
        response = requests.get("https://serpapi.com/search", params=params).json()
        batch = response.get("reviews", [])

        #If the batch is empty then all of the reviews have been collected
        if not batch:
            print("No more reviews.")
            break

        # Adds the page's reviews to the master list
        #Takes the next page token from the pagination section, stops the loop if last page is reached
        all_reviews.extend(batch)
        print(f"Collected {len(all_reviews)} Google reviews")

        next_page_token = response.get("serpapi_pagination", {}).get("next_page_token")
        if not next_page_token:
            break

     # Write the collected reviews to a JSON file
    os.makedirs("data/raw", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_reviews[:limit], f, indent=4)
    print(f"Saved to {filename}")


def fetch_yelp_reviews(place_id, filename, limit=400):
    #Using a list for all accumulated reviews and fetching the next page of results
    print("Fetching Yelp reviews.")
    all_reviews = []
    start = 0

    while len(all_reviews) < limit:
        # Build the query parameters for the SerpApi Yelp Reviews engine
        # Has a unique Yelp business ID for the restaurant
        params = {
            "api_key": API_KEY,
            "engine": "yelp_reviews",
            "place_id": place_id,
            "start": start,
            "num": 49, #number of reviews to return per page
        }
        # Makes the HTTP GET request to SerpApi and parse the JSON results
        response = requests.get("https://serpapi.com/search", params=params).json()
        batch = response.get("reviews", [])

        if not batch:
            print("No more reviews.")
            break

        all_reviews.extend(batch)
        print(f"Collected {len(all_reviews)} Yelp reviews")
        start += 49

    # Write the collected reviews to a JSON file
    os.makedirs("data/raw", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_reviews[:limit], f, indent=4)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    # Fetch Yelp reviews using the restaurant's unique Yelp place_id
    fetch_yelp_reviews(
        place_id="auitSrx5FpFLkfHKrksYyg",
        filename="data/raw/yelp_raw.json"
    )
    # Fetch Google Maps reviews using the restaurant's unique Maps data_id
    fetch_google_reviews(
        data_id="0x80c2908a16708b21:0x62870c3bdfd1c32c",
        filename="data/raw/google_raw.json"
    )
