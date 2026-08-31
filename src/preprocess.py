import pandas as pd
import json
import os
import re

#Loads and parses the Google reviews from the JSON
# Extracted the text, rating, date, and dining details.
def load_google_reviews(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for review in data:
        details = review.get("details", {})
        text = review.get("snippet") or review.get("extracted_snippet", {}).get("original", "")
        rows.append({
            "source": "Google",
            "rating": review.get("rating"),
            "text": text,
            "date": review.get("iso_date", ""),
            "order_type": details.get("order_type"),
            "meal_type": details.get("meal_type"),
            "price_per_person": details.get("price_per_person"),
            "food_rating": details.get("food"),
            "service_rating": details.get("service"),
            "atmosphere_rating": details.get("atmosphere"),
            "recommended_dishes": details.get("recommended_dishes"),
            "wait_time": details.get("wait_time"),
        })
    
    return pd.DataFrame(rows)

#Loads and parses Yelp reviews from raw JSON
# Extracted the text, rating, date, and user metadata.
def load_yelp_reviews(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for review in data:
        user = review.get("user", {})
        feedback = review.get("feedback", {})
        rows.append({
            "source": "Yelp",
            "rating": review.get("rating"),
            "text": review.get("comment", {}).get("text", ""),
            "date": review.get("date", ""),
            "user_location": user.get("address"),
            "user_reviews": user.get("reviews"),
            "user_elite": user.get("elite_year"),
            "feedback_useful": feedback.get("useful"),
            "feedback_funny": feedback.get("funny"),
            "feedback_cool": feedback.get("cool"),
        })
    
    return pd.DataFrame(rows)

#Strips the whitespace, collapses newlines, and returns None for non-string or short text.
def clean_text(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    if len(text) < 10:
        return None
    return text

#Combines both the Google and Yelp reviews into a cleaned DataFrame and saved to a CSV.
def preprocess():
    google_df = load_google_reviews("data/raw/google_raw.json")
    yelp_df = load_yelp_reviews("data/raw/yelp_raw.json")
    
    df = pd.concat([google_df, yelp_df], ignore_index=True)
    df['text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['text', 'rating'])
    df = df[df['text'].str.strip() != '']
    df = df.reset_index(drop=True)
    
    assert df['text'].isna().sum() == 0
    assert (df['text'] == '').sum() == 0

    
    print(f"Total rows: {len(df)}")
    print(f"Google: {len(df[df['source'] == 'Google'])}")
    print(f"Yelp: {len(df[df['source'] == 'Yelp'])}")
    print(f"\nRating distribution:\n{df['rating'].value_counts().sort_index()}")
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed_reviews.csv", index=False)
    print("Saved to data/processed/processed_reviews.csv")


if __name__ == "__main__":
    preprocess()


