import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from umap import UMAP
from PIL import Image
import textwrap
import seaborn as sns

def make_train_valid_dfs():
    dataframe = pd.read_csv("./data/medical/merged_dataset_test.csv")
    return dataframe.reset_index(drop=True)

def get_image_embeddings(valid_df):
    image_folder_path = "./data/medical/test/"
    images = []
    target_size = (224, 224)

    for image_path in valid_df["image"]:
        img_path = os.path.join(image_folder_path, f"{image_path}.jpg")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        if image is not None:
            image = cv2.resize(image, target_size) 
            images.append(image)

    images = np.array(images)
    num_images, height, width = images.shape
    flattened_images = images.reshape(num_images, height * width)
    return flattened_images

def cluster_embeddings(embeddings, num_clusters):
    clustering_kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = clustering_kmeans.fit_predict(embeddings)
    return cluster_labels

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

def main():
    valid_df = make_train_valid_dfs()
    image_embeddings = get_image_embeddings(valid_df)

    # Reduce dimensionality using UMAP
    umap_model = UMAP(n_components=5)
    reduced_embeddings = umap_model.fit_transform(image_embeddings)
    print(reduced_embeddings.shape)

    num_clusters = 6 
    labels = cluster_embeddings(reduced_embeddings, num_clusters)

    umap_model_2d = UMAP(n_components=2)
    reduced_embeddings_2d = umap_model_2d.fit_transform(reduced_embeddings)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(np.unique(labels)))
    sns.scatterplot(x=reduced_embeddings_2d[:, 0], y=reduced_embeddings_2d[:, 1], hue=labels, palette=palette, legend='full')
    plt.title('UMAP projection of image embeddings with KMeans clusters', fontname='Corbel', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontname='Corbel', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontname='Corbel', fontsize=12)
    plt.legend(title='Cluster', fontsize=10, title_fontsize='13', loc='best')
    plt.show()

    # Compute clustering evaluation metrics
    silhouette_avg = silhouette_score(reduced_embeddings, labels)
    print(f'Silhouette Score: {silhouette_avg}')

    ch_score = calinski_harabasz_score(reduced_embeddings, labels)
    print(f'Calinski-Harabasz Index: {ch_score}')

    db_score = davies_bouldin_score(reduced_embeddings, labels)
    print(f'Davies-Bouldin Index: {db_score}')

    visualize_cluster_samples(valid_df, labels)

    # Elbow method for optimal k
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

    # Silhouette scores for different numbers of clusters
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
    print(silhouette_scores)
    
    # KElbowVisualizer for optimal k
    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(2, 21))
    visualizer.fit(reduced_embeddings)
    visualizer.show()

    # Silhouette analysis for different numbers of clusters
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