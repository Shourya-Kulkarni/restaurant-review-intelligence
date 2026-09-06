import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

#Reduces embedding dimensions with UMAP then apply KMeans clustering to group similar reviews.
#Iterated through different numbers of clusters to find the best silhoutte score (a score of how well the clusters are sectioned) and landed on k=3
def cluster_reviews(embeddings_path, reviews_path, output_path, n_clusters=3):
    
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(reviews_path)
    
    reducer = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    print(f"Reduced to shape: {reduced.shape}")
    

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(reduced)
    
    score = silhouette_score(reduced, df['cluster'])
    print(f"Silhouette score: {score:.4f}")
    
    print(f"\nCluster distribution:\n{df['cluster'].value_counts().sort_index()}")
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    cluster_reviews(
        embeddings_path="artifacts/review_embeddings.npy",
        reviews_path="data/processed/processed_reviews.csv",
        output_path="data/processed/clustered_reviews.csv",
        n_clusters=3
    )