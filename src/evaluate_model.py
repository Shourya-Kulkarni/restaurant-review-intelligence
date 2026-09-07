import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Manually added labels to complaints to validate my clusters against human ground truth
#3 types of complaints : Service (S), Food (F), Operations/Logistics/Pricing (O)
manual_labels = {
    0: 'S', 1: 'F', 2: 'O', 3: 'S', 4: 'S',
    5: 'S', 6: 'O', 7: 'O', 8: 'F', 9: 'S',
    10: 'F', 11: 'S', 12: 'S', 13: 'F', 14: 'F',
    15: 'F', 16: 'F', 17: 'S', 18: 'S', 19: 'S',
    20: 'F', 21: 'F', 22: 'F', 23: 'S', 24: 'F'
}
df = pd.read_csv('data/processed/clustered_reviews.csv')
df['human_label'] = df.index.map(manual_labels)

# Maps KMeans clusters to theme names based on what I observed
# Compared KMeans clusters against human ground truth labels across 3 complaint categories, achieving 80% accuracy and 0.83 weighted F1-score.
# Operational failures like incorrect hours and pricing surprises were linguistically framed as service complaints, causing 4 misclassifications
# A possible recommendation to the client: fixing operational gaps would likely reduce perceived service complaints without any staff training required.
cluster_map = {0: 'S', 1: 'F', 2: 'O'}
df['kmeans_label'] = df['cluster'].map(cluster_map)

print(classification_report(df['human_label'], df['kmeans_label']))
print(confusion_matrix(df['human_label'], df['kmeans_label']))