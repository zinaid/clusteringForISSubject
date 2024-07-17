import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

import torch
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import seaborn as sns
from PIL import Image
import textwrap
import config as CFG
from main import build_loaders
from CLIP import CLIPModel

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

def cluster_embeddings(embeddings, num_clusters):
    clustering_kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = clustering_kmeans.fit_predict(embeddings)
    return cluster_labels

def visualize_cluster_samples(valid_df, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    _, axes = plt.subplots(len(unique_clusters), 3, figsize=(15, 10))

    for i, cluster in enumerate(unique_clusters):
        cluster_samples = valid_df[cluster_labels == cluster].head(3)
        for j, (_, row) in enumerate(cluster_samples.iterrows()):
            image_path = row["image"]
            image_id = row["id"]
            image = cv2.imread(f"./data/medical/test/{image_path}.jpg")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(image)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"Cluster: {cluster}\nID: {image_id}")
    
    plt.tight_layout()
    plt.show()

def visualize_cluster_samples(valid_df, cluster_labels):
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(idx)
    
    num_samples_per_cluster = 9
    for cluster, indices in clusters.items():
        cluster_samples = np.random.choice(indices, num_samples_per_cluster, replace=False)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i, sample_idx in enumerate(cluster_samples):
            image_path = valid_df.iloc[sample_idx]["image"]
            image_id = valid_df.iloc[sample_idx]["id"]
            caption = valid_df.iloc[sample_idx]["caption"]
            modality = valid_df.iloc[sample_idx]["modality"]
            image = Image.open(f"./data/medical/test/{image_path}.jpg").convert("RGB")
            axes[i].imshow(image)
            axes[i].axis("off")
            axes[i].set_title(f"Modality: {modality}", fontsize=8)
            
            wrapped_caption = textwrap.fill(caption, 50)  
            axes[i].text(1.1, 0.5, wrapped_caption, size=8, ha="left", va="center", transform=axes[i].transAxes)

        fig.suptitle(f"Cluster {cluster} Visualization", fontsize=12)
        plt.tight_layout()
        plt.show()

def make_train_valid_dfs():
    dataframe = pd.read_csv("./data/medical/merged_dataset_test.csv")
    return dataframe.reset_index(drop=True)

def main():
    valid_df = make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(valid_df, "./model_checkpoints/bestmed.pt")
    
    from umap import UMAP

    umap_model = UMAP(n_components=5)
    reduced_embeddings = umap_model.fit_transform(image_embeddings.cpu().numpy())
    print(reduced_embeddings.shape) 

    clustering_kmeans = KMeans(n_clusters=10, random_state=42)
    clustering_kmeans.fit(reduced_embeddings)
    labels = clustering_kmeans.labels_

    umap_model_2d = UMAP(n_components=2)
    reduced_embeddings_2d = umap_model_2d.fit_transform(reduced_embeddings)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(np.unique(labels)))
    sns.scatterplot(x=reduced_embeddings_2d[:, 0], y=reduced_embeddings_2d[:, 1], hue=labels, palette=palette, legend='full')
    plt.title('UMAP projection of CLIP embeddings with KMeans clusters', fontname='Corbel', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontname='Corbel', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontname='Corbel', fontsize=12)
    plt.legend(title='Cluster', fontsize=10, title_fontsize='13', loc='best')
    plt.show()

    silhouette_avg = silhouette_score(reduced_embeddings, labels)
    print(f'Silhouette Score: {silhouette_avg}')

    ch_score = calinski_harabasz_score(reduced_embeddings, labels)
    print(f'Calinski-Harabasz Index: {ch_score}')

    db_score = davies_bouldin_score(reduced_embeddings, labels)
    print(f'Davies-Bouldin Index: {db_score}')

    visualize_cluster_samples(valid_df, labels)

    inertia = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(reduced_embeddings)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 20), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    silhouette_scores = []
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(reduced_embeddings)
        silhouette_avg = silhouette_score(reduced_embeddings, clusters)
        silhouette_scores.append(silhouette_avg)

    plt.plot(range(2, 21), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.show()

    for k, score in zip(range(2, 21), silhouette_scores):
        print(f'Silhouette Score for {k} clusters: {score}')

    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(2, 21))
    visualizer.fit(reduced_embeddings)
    visualizer.show()

    num_clusters = 12
    num_rows = (num_clusters - 1) // 2 + 1
    fig, ax = plt.subplots(num_rows, 2, figsize=(15, 6 * num_rows))
    fig.suptitle('Silhouette Analysis for Different Numbers of Clusters', fontsize=16)

    for i in range(2, num_clusters + 1):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i - 2, 2)
        
        if q < num_rows:
            visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q][mod])
            visualizer.fit(reduced_embeddings)
            visualizer.ax.set_title(f'{i} Clusters')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
