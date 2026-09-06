import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os


#Loads the reviews and generates embeddings using a pretrained transformer
#Saved as a numpy array
def generate_embeddings(input_filepath, output_filepath):    
    df = pd.read_csv(input_filepath)
    texts = df['text'].tolist()
    
    print(f"Generating embeddings for {len(texts)} reviews...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    
    os.makedirs("artifacts", exist_ok=True)
    np.save(output_filepath, embeddings)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved to {output_filepath}")


if __name__ == "__main__":
    generate_embeddings(
        input_filepath="data/processed/processed_reviews.csv",
        output_filepath="artifacts/review_embeddings.npy"
    )