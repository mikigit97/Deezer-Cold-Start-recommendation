import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
from options import config
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def train_kmeans(dataset_path, master_path, clustering_path, nb_clusters, max_iter, random_state, 
                 embeddings_version="svd", clusters_filename=None):
    # Set default filename if not provided
    if clusters_filename is None:
        clusters_filename = "kmeans_model_embeddings.pkl"

    # ---------------------------
    # 1. Load User Embeddings
    # ---------------------------
    user_embeddings = pd.read_parquet(os.path.join(dataset_path, "user_embeddings.parquet"), engine='fastparquet')
    # Expand the embeddings list into individual columns
    list_embeddings = ["embedding_" + str(i) for i in range(len(user_embeddings[embeddings_version + "_embeddings"].iloc[0]))]
    user_embeddings[list_embeddings] = pd.DataFrame(user_embeddings[embeddings_version + "_embeddings"].tolist(),
                                                    index=user_embeddings.index)
    user_embeddings_values = user_embeddings[list_embeddings].values

    # ---------------------------
    # 2. Run KMeans Clustering
    # ---------------------------
    # kmeans = KMeans(n_clusters=nb_clusters, random_state=random_state, max_iter=max_iter,
    #                 algorithm='lloyd', n_init='auto').fit(user_embeddings_values)
    # load interaction counts and build per‐user weights in [1,4]

    '''Experiment weighted warm user clustering'''

    # with open('/home/mickaelz/Recommendation systems/project/user_interactions_count.pkl','rb') as wf:
    #     interactions = pickle.load(wf)
    # # map to dataframe order
    # counts = user_embeddings['user_index'].map(interactions).fillna(0).astype(float)
    # cmin, cmax = counts.min(), counts.max()
    # if cmax>cmin:
    #     weights = 1.0 + 2.0*(counts - cmin)/(cmax - cmin)
    # else:
    #     weights = pd.Series(1.0, index=counts.index)

    # kmeans = KMeans(n_clusters=nb_clusters,
    #             random_state=random_state,
    #             max_iter=max_iter,
    #             algorithm='lloyd',
    #             n_init='auto') \
    #         .fit(user_embeddings_values,
    #             sample_weight=weights.values)

    # Save the KMeans model
    # model_dir = os.path.join(master_path, clustering_path)
    # os.makedirs(model_dir, exist_ok=True)
    # with open(os.path.join(model_dir, clusters_filename), "wb") as f:
    #     pickle.dump(kmeans, f)
    # with open('/home/mickaelz/Recommendation systems/project/user_interactions_count.pkl', 'rb') as wf:
    #     interactions = pickle.load(wf)
    # # map to dataframe order
    # counts = user_embeddings['user_index'].map(interactions).fillna(0).astype(float)
    # cmin, cmax = counts.min(), counts.max()
    # if cmax > cmin:
    #     # Scale counts directly to [0.0, 1.0] range
    #     weights = (counts - cmin) / (cmax - cmin)
    # else:
    #     weights = pd.Series(0.5, index=counts.index)  # Default mid-weight if all counts are equal

    # -- Evaluate cluster sizes for optimal number using inertia --
    # eval_results = []
    # eval_model_dir = os.path.join(master_path, clustering_path)
    # os.makedirs(eval_model_dir, exist_ok=True)
    # for k in range(500, 1201, 100):
    #     km_tmp = KMeans(n_clusters=k,
    #                     random_state=random_state,
    #                     max_iter=max_iter,
    #                     algorithm='lloyd',
    #                     n_init='auto')
    #     km_tmp.fit(user_embeddings_values)
    #     eval_results.append({"n_clusters": k, "inertia": km_tmp.inertia_})
    # eval_df = pd.DataFrame(eval_results)
    # eval_dir = os.path.join(eval_model_dir, "evaluation")
    # os.makedirs(eval_dir, exist_ok=True)
    # eval_df.to_csv(os.path.join(eval_dir, "clusters_evaluation.csv"), index=False)
    # print(f"Cluster evaluation saved to {os.path.join(eval_dir, 'clusters_evaluation.csv')}")

    # kmeans = KMeans(n_clusters=nb_clusters,
    #             random_state=random_state,
    #             max_iter=max_iter,
    #             algorithm='lloyd',
    #             n_init='auto') \
    #         .fit(user_embeddings_values)
    '''-- Evaluate cluster sizes for optimal number using inertia --'''
    eval_results = []
    eval_model_dir = os.path.join(master_path, clustering_path)
    os.makedirs(eval_model_dir, exist_ok=True)
    for k in range(500, 2001, 100):
        km_tmp = KMeans(n_clusters=k,
                        random_state=random_state,
                        max_iter=max_iter,
                        algorithm='lloyd',
                        n_init='auto')
        km_tmp.fit(user_embeddings_values)
        eval_results.append({"n_clusters": k, "inertia": km_tmp.inertia_})
    eval_df = pd.DataFrame(eval_results)
    eval_dir = os.path.join(eval_model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    eval_df.to_csv(os.path.join(eval_dir, "clusters_evaluation.csv"), index=False)
    print(f"Cluster evaluation saved to {os.path.join(eval_dir, 'clusters_evaluation.csv')}")

    # -- Automatic elbow detection --
    ks = eval_df['n_clusters'].tolist()
    inertias = eval_df['inertia'].tolist()
    try:
        from kneed import KneeLocator
        knee = KneeLocator(ks, inertias,
                           curve='convex',
                           direction='decreasing').knee
        optimal_k = int(knee or nb_clusters)
    except ImportError:
        print("kneed not installed, falling back to nb_clusters")
        optimal_k = nb_clusters

    # -- Plot elbow curve --
    plt.figure()
    plt.plot(ks, inertias, 'o-', label='inertia')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    if optimal_k in ks:
        ymin, ymax = min(inertias), max(inertias)
        plt.vlines(optimal_k, ymin, ymax,
                   linestyles='dashed',
                   color='red',
                   label=f'optimal k = {optimal_k}')
    plt.legend()
    elbow_file = os.path.join(eval_dir, "elbow_plot.png")
    plt.savefig(elbow_file, dpi=150)
    plt.close()
    print(f"Elbow plot saved to {elbow_file}")

    # -- Final KMeans with optimal k --
    kmeans = KMeans(n_clusters=1000,
                    random_state=random_state,
                    max_iter=max_iter,
                    algorithm='lloyd',
                    n_init='auto') \
                .fit(user_embeddings_values)
    
    #   kmeans = KMeans(n_clusters=nb_clusters, random_state=random_state, max_iter=max_iter,
    #                 algorithm='lloyd', n_init='auto').fit(user_embeddings_values)

    # Save the KMeans model
    model_dir = os.path.join(master_path, clustering_path)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, clusters_filename), "wb") as f:
        pickle.dump(kmeans, f)

       # --- Plot clusters in 2D via PCA ---
    # pca = PCA(n_components=2)
    # X2 = pca.fit_transform(user_embeddings_values)
    # plt.figure(figsize=(8,6))
    # pts = plt.scatter(X2[:,0], X2[:,1],
    #                     c=kmeans.labels_,
    #                     cmap='tab10',
    #                     s=15, alpha=0.7)
    # plt.title(f"KMeans ({nb_clusters} clusters) in 2D PCA")
    # plt.xlabel("PC1"); plt.ylabel("PC2")
    # plt.colorbar(pts, label="cluster")
    # plt.tight_layout()
    # plot_file = os.path.join(model_dir, "clusters_plot.png")
    # plt.savefig(plot_file, dpi=150)
    # plt.close()
    # print(f"Cluster plot saved to {plot_file}")
    # --- Plot clusters in a more informative way ---
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(user_embeddings_values)
    plt.figure(figsize=(16, 12))

    # 1. Main plot: Show data with centroids
    plt.subplot(2, 2, 1)
    plt.scatter(X2[:, 0], X2[:, 1], c='lightgray', s=10, alpha=0.1)
    centroids_2d = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                c='red', s=30, marker='x')
    plt.title(f"KMeans Centroids ({nb_clusters} clusters)")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    # 2. Density plot: Show data density
    plt.subplot(2, 2, 2)
    plt.hist2d(X2[:, 0], X2[:, 1], bins=100, cmap='viridis')
    plt.colorbar(label="Count")
    plt.title("Data Density")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    # 3. Sample of clusters: Show random 10 clusters only
    plt.subplot(2, 2, 3)
    random_clusters = np.random.choice(nb_clusters, size=10, replace=False)
    mask = np.isin(kmeans.labels_, random_clusters)
    plt.scatter(X2[mask, 0], X2[mask, 1], 
                c=kmeans.labels_[mask], cmap='tab10', 
                s=20, alpha=0.7)
    plt.title("Sample of 10 Random Clusters")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    # 4. Weight visualization: Show points sized by their weights
    plt.subplot(2, 2, 4)
    plt.scatter(X2[:, 0], X2[:, 1], 
                c=kmeans.labels_, cmap='tab10', alpha=0.2,)  # Size by weight
    plt.title("Points Sized by Interaction Weight")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    plt.tight_layout()
    plot_file = os.path.join(model_dir, "clusters_analysis.png")
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Enhanced cluster analysis saved to {plot_file}")

    # ---------------------------
    # 3. Assign Clusters and Prepare Song Aggregation
    # ---------------------------
    user_embeddings["cluster"] = kmeans.labels_
    user_clusters = user_embeddings[["user_index", "cluster"]]

    # Load training user features (songs listened D1-D30)
    features_train_path = os.path.join(dataset_path, "user_features_train_" + embeddings_version + ".parquet")
    features_train = pd.read_parquet(features_train_path, engine='fastparquet').fillna(0)
    features_train = features_train.sort_values("user_index").reset_index(drop=True)

    # Merge user clusters with song lists
    listd1d30 = pd.merge(features_train[["user_index", "d1d30_songs"]], user_clusters, on="user_index", how="left")
    listd1d30_exploded = listd1d30.explode('d1d30_songs')
    listd1d30_exploded["count"] = 1  # each occurrence counts as 1

    # Group by cluster and song to get counts
    listd1d30_by_cluster = pd.DataFrame(listd1d30_exploded.groupby(["cluster", "d1d30_songs"])['count'].count())

    # ---------------------------
    # 4. Ensure All Cluster–Song Combinations Exist
    # ---------------------------
    nb_songs = config["nb_songs"]
    # Create the complete MultiIndex for (cluster, song_index)
    complete_index = pd.MultiIndex.from_product([np.arange(nb_clusters), np.arange(nb_songs)],
                                                  names=["cluster", "song_index"])
    # Reindex the grouped counts to include all combinations, filling missing values with 0
    both = listd1d30_by_cluster.reindex(complete_index, fill_value=0).reset_index()
    both.columns = ["cluster", "song_index", "nb_streams"]

    # ---------------------------
    # 5. Compute Song Probabilities and Save Them
    # ---------------------------
    data_by_cluster = both.groupby("cluster")["nb_streams"].sum().rename("nb_streams_by_cluster").reset_index()
    data_by_cluster_and_song = pd.merge(both, data_by_cluster, on="cluster")
    data_by_cluster_and_song["segment_proba"] = data_by_cluster_and_song["nb_streams"] / data_by_cluster_and_song["nb_streams_by_cluster"]

    probas_dir = os.path.join(master_path, clustering_path + "_probas_" + embeddings_version)
    os.makedirs(probas_dir, exist_ok=True)

    for cluster_id in range(nb_clusters):
        if cluster_id % 100 == 0:
            print("Song probabilities computed for cluster:", cluster_id)
        cluster_df = data_by_cluster_and_song[data_by_cluster_and_song["cluster"] == cluster_id].sort_values("song_index")
        # Select top nb_songs probabilities (if there are fewer than nb_songs, pad with 0)
        proba_list = cluster_df["segment_proba"].tolist()
        if len(proba_list) < nb_songs:
            proba_list.extend([0.0] * (nb_songs - len(proba_list)))
        else:
            proba_list = proba_list[:nb_songs]
        with open(os.path.join(probas_dir, f"list_proba_{cluster_id}.pkl"), "wb") as f:
            pickle.dump(proba_list, f)


def train_inputfeatureskmeans(dataset_path, master_path, clustering_path, nb_clusters, max_iter,
                              random_state, nb_songs, embeddings_version="svd", clusters_filename=None):
    # Set default filename if not provided
    if clusters_filename is None:
        clusters_filename = "kmeans_model_inputfeatures.pkl"

    # ---------------------------
    # 1. Load and Normalize Input Features
    # ---------------------------
    user_features_train = pd.read_parquet(os.path.join(dataset_path, "user_features_train_" + embeddings_version + ".parquet"),
                                          engine='fastparquet')
    features_train = user_features_train.fillna(0).sort_values("user_index")
    # Assume features are from column index 2 onward (adjust if necessary)
    features_train_arr = features_train.values[:, 2:]
    transformer = Normalizer().fit(features_train_arr)
    X_train = transformer.transform(features_train_arr)

    # ---------------------------
    # 2. Run KMeans Clustering on Input Features
    # ---------------------------
    kmeans = KMeans(n_clusters=nb_clusters, random_state=random_state, max_iter=max_iter,
                    algorithm='lloyd', n_init='auto').fit(X_train)
    model_dir = os.path.join(master_path, clustering_path)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, clusters_filename), "wb") as f:
        pickle.dump(kmeans, f)

    # ---------------------------
    # 3. Assign Clusters to Training Data
    # ---------------------------
    features_train["cluster"] = kmeans.labels_
    listd1d30 = features_train[["user_index", "d1d30_songs", "cluster"]]
    listd1d30_exploded = listd1d30.explode("d1d30_songs")
    listd1d30_exploded["count"] = 1
    listd1d30_by_cluster = pd.DataFrame(listd1d30_exploded.groupby(["cluster", "d1d30_songs"])["count"].count())

    # ---------------------------
    # 4. Ensure All Cluster–Song Combinations Exist
    # ---------------------------
    arrays = (np.repeat(np.arange(nb_clusters), nb_songs),
              np.tile(np.arange(nb_songs), nb_clusters))
    complete_index = pd.MultiIndex.from_product([np.arange(nb_clusters), np.arange(nb_songs)],
                                                  names=["cluster", "song_index"])
    both = listd1d30_by_cluster.reindex(complete_index, fill_value=0).reset_index()
    both.columns = ["cluster", "song_index", "nb_streams"]
    both = both.sort_values(["cluster", "song_index"]).reset_index(drop=True)

    data_by_cluster = both.groupby("cluster")["nb_streams"].sum().rename("nb_streams_by_cluster").reset_index()
    data_by_cluster_and_song = pd.merge(both, data_by_cluster, on="cluster")
    data_by_cluster_and_song["segment_proba"] = data_by_cluster_and_song["nb_streams"] / data_by_cluster_and_song["nb_streams_by_cluster"]

    # ---------------------------
    # 5. Save Top Song Probabilities per Cluster
    # ---------------------------
    probas_dir = os.path.join(master_path, clustering_path + "_probas_" + embeddings_version)
    os.makedirs(probas_dir, exist_ok=True)

    for cluster_id in range(nb_clusters):
        if cluster_id % 10 == 0:
            print("Song probabilities computed for cluster:", cluster_id)
        cluster_df = data_by_cluster_and_song[data_by_cluster_and_song["cluster"] == cluster_id].sort_values("song_index")
        proba_list = cluster_df["segment_proba"].tolist()
        if len(proba_list) < nb_songs:
            proba_list.extend([0.0] * (nb_songs - len(proba_list)))
        else:
            proba_list = proba_list[:nb_songs]
        with open(os.path.join(probas_dir, f"list_proba_{cluster_id}.pkl"), "wb") as f:
            pickle.dump(proba_list, f)
