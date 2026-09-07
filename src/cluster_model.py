import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import joblib
import os

#Since there were so many positive reviews (473 out of 564), I took an approach of only filtering and clustering negative reviews (91 total)
#As a way to provide genuine actionable insights to my client
#Chose k=3 because it cleanly splits complaints into three actionable pillars without over-fragmenting
def cluster_negative_reviews(reviews_path, output_path, n_clusters=3):
    
    df = pd.read_csv(reviews_path)

    #The restaurant had moved locations, so I filtered the reviews for when they opened in their new location for relevancy
    #There were only 25 negative reviews in their new location
    df['date'] = pd.to_datetime(df['date'], utc=True)
    cutoff_date = pd.to_datetime('2021-09-01', utc=True)
    df_sepulveda = df[df['date'] >= cutoff_date].reset_index(drop=True)

    negative_df = df_sepulveda[df_sepulveda['rating'] <= 3].reset_index(drop=True)
    positive_df = df[df['rating'] >= 4].reset_index(drop=True)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(negative_df['text'].tolist(), show_progress_bar=True)
    
    reducer = UMAP(n_components=5, n_neighbors=10, min_dist=0.0, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    # Save 2D UMAP coordinates for visualization
    reducer_2d = UMAP(n_components=2, n_neighbors=10, min_dist=0.0, random_state=42)
    reduced_2d = reducer_2d.fit_transform(embeddings)
    negative_df['umap_x'] = reduced_2d[:, 0]
    negative_df['umap_y'] = reduced_2d[:, 1]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    negative_df['cluster'] = kmeans.fit_predict(reduced)
    
    score = silhouette_score(reduced, negative_df['cluster'])
    print(f"Silhouette score: {score:.4f}")
    print(f"\nCluster distribution:\n{negative_df['cluster'].value_counts().sort_index()}")
    
    # Print sample reviews per cluster
    for cluster_id in range(n_clusters):
        print(f"\n=== COMPLAINT CLUSTER {cluster_id} ===")
        samples = negative_df[negative_df['cluster'] == cluster_id]['text'].head(3).tolist()
        for s in samples:
            print(f"- {s[:200]}")
    
    os.makedirs("data/processed", exist_ok=True)
    negative_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(reducer, "artifacts/umap_reducer.joblib")
    joblib.dump(kmeans, "artifacts/kmeans_model.joblib")
    print("Saved artifacts/umap_reducer.joblib and artifacts/kmeans_model.joblib")

    # Compute per-cluster centroids in the original 384-dim embedding space.
    # Used for stable cosine-similarity prediction on unseen text (UMAP is visualization-only).
    full_embeddings = model.encode(negative_df['text'].tolist(), show_progress_bar=False)
    centroids = np.array([
        full_embeddings[negative_df['cluster'].values == k].mean(axis=0)
        for k in range(n_clusters)
    ])
    np.save("artifacts/cluster_centroids.npy", centroids)
    print(f"Saved artifacts/cluster_centroids.npy  shape={centroids.shape}")


if __name__ == "__main__":
    cluster_negative_reviews(
        reviews_path="data/processed/processed_reviews.csv",
        output_path="data/processed/clustered_reviews.csv",
        n_clusters=3
    )