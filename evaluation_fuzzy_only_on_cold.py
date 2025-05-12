import os
import pandas as pd
import numpy as np
import torch
import torch.nn
import pickle
from model import RegressionTripleHidden
# from model import EmbeddingRegressor128
from options import config
from sklearn.metrics import ndcg_score, dcg_score
import statistics
from sklearn.preprocessing import Normalizer
import itertools
import time



# ---------------------------
# Main Evaluation Function with Hard-Clustering Based Fuzzy Recommendation for Cold Users
# ---------------------------
def evaluation_with_fuzzy_on_cold(dataset_path, master_path, eval_type="full_perso", embeddings_version="svd", 
                          model_filename=None, clustering_path=None, clusters_filename=None, 
                          nb_clusters=config["nb_clusters"]):
    print("=== Starting Evaluation ===")
    use_cuda = config['use_cuda']
    target_dim = config['embeddings_dim']
    input_dim = config['input_dim']
    k_val_list = config["k_val_list"]
    indic_eval_evolution = config["indic_eval_evolution"]
    cuda = torch.device(0)
    model_filename = os.path.join(master_path, model_filename + ".pt")
    
    # ---------------------------
    # 1. Load Testing Dataset
    # ---------------------------
    test_dir = os.path.join(master_path, embeddings_version, "test")
    print("--- Loading testing dataset from:", test_dir)
    testing_set_size = int(len(os.listdir(test_dir)) / 3)
    print(f"Found {testing_set_size} test samples.")
    
    test_xs = []
    listened_songs_test_ys = []
    groundtruth_list_test = []
    for idx in range(testing_set_size):
        if eval_type in ["full_perso", "semi_perso", "popularity"]:
            # if idx % 100 == 0:
            #     print(f"Loading test sample {idx}...")
            x = pickle.load(open(os.path.join(test_dir, f"x_{idx}.pkl"), "rb"))
            test_xs.append(x)
        elif eval_type in ["inputfeatures"]:
            vector = pickle.load(open(os.path.join(test_dir, f"x_{idx}.pkl"), "rb"))
            transformer = Normalizer().fit(vector.reshape(1, -1))
            norm_vector = torch.FloatTensor(transformer.transform(vector.reshape(1, -1))[0])
            test_xs.append(norm_vector)
        y = pickle.load(open(os.path.join(test_dir, f"y_listened_songs_{idx}.pkl"), "rb"))
        gt = pickle.load(open(os.path.join(test_dir, f"groundtruth_list_{idx}.pkl"), "rb"))
        listened_songs_test_ys.append(y)
        groundtruth_list_test.append(gt)
    
    if eval_type in ["avgd0stream"]:
        print("Loading average d0 stream data for testing...")
        df = pd.read_parquet(os.path.join(dataset_path, f"user_features_test_{embeddings_version}.parquet"), engine='fastparquet')
        cols = list(df)[2+target_dim*10:2+target_dim*10+target_dim]
        avgd0stream_df = df[["user_index"] + cols].set_index("user_index").sort_index()
        test_xs = avgd0stream_df.values
    print("Loaded testing dataset.")
    total_test_dataset = list(zip(test_xs, listened_songs_test_ys, groundtruth_list_test))
    print(f"Total test dataset size: {len(total_test_dataset)} samples.")
    if len(total_test_dataset) > 0:
        sample_shape = total_test_dataset[0][0].shape if hasattr(total_test_dataset[0][0], "shape") else "N/A"
        print(f"Shape of first test sample: {sample_shape}")
    
    # ---------------------------
    # 2. Load Song Embeddings and Hard-Clustering Model Info
    # ---------------------------
    # Check if the evaluation type requires loading song embeddings
    if eval_type in ["full_perso", "semi_perso", "avgd0stream"]:
        # Print the path of the song embeddings file being loaded
        print("--- Loading song embeddings from:", os.path.join(dataset_path, "song_embeddings.parquet"))
        
        # Load the song embeddings from a parquet file and fill any missing values with 0
        song_embeddings = pd.read_parquet(os.path.join(dataset_path, "song_embeddings.parquet"), engine='fastparquet').fillna(0)
        
        # Generate a list of feature column names based on the embedding version
        list_features = ["feature_" + str(i) for i in range(len(song_embeddings["features_" + embeddings_version][0]))]
        
        # Use the column numbers to separate the song_embedding values by column
        song_embeddings[list_features] = pd.DataFrame(song_embeddings["features_" + embeddings_version].tolist(), 
                                                       index=song_embeddings.index)
        
        # Convert the embedding features into a NumPy array
        song_embeddings_values = song_embeddings[list_features].values
        
        # Convert the NumPy array of embedding features into a PyTorch FloatTensor
        song_embeddings_values_ = torch.FloatTensor(song_embeddings_values.astype(np.float32))
        print(f"Loaded {len(song_embeddings_values_)} song embeddings.")
        
        if eval_type in ["full_perso", "semi_perso"]:
            print("--- Loading regression model from:", model_filename)
            regression_model = RegressionTripleHidden(input_dim=input_dim, output_dim=target_dim)
            # regression_model = EmbeddingRegressor128()
            regression_model.load_state_dict(torch.load(model_filename))
            reg = regression_model.eval()
            if use_cuda:
                reg = reg.to(device=cuda)
            print("Regression model loaded:\n", reg)
            
        if eval_type in ["semi_perso"]:
            # For hard clustering, we load the KMeans model (instead of a fuzzy model)
            print("--- Loading KMeans model for semi_perso evaluation ---")
            kmeans_model_path = os.path.join(master_path, clustering_path, clusters_filename)
            with open(kmeans_model_path, "rb") as f:
                kmeans = pickle.load(f)
                # support both sklearn.KMeans and your fuzzy dict
            if "fuzzy" in kmeans_model_path:
                centers = kmeans["cluster_centers_"]
            else:
                centers = kmeans.cluster_centers_
            if use_cuda:
                centroid_ = torch.FloatTensor(centers).to(device=cuda)
            else:
                centroid_ = torch.FloatTensor(centers)
            print(f"Loaded KMeans model with {len(centroid_)} centroids.")
            
            print("--- Loading song probability lists for each cluster ---")
            song_proba_by_segment = []
            for cluster_id in range(nb_clusters):
                path = os.path.join(master_path, f"{clustering_path}_probas_{embeddings_version}", f"list_proba_{cluster_id}.pkl")
                song_proba_by_segment.append(pickle.load(open(path, "rb")))
            print(f"Loaded song probabilities for {len(song_proba_by_segment)} clusters.")
    
    elif eval_type in ["popularity"]:
        print("--- Loading popularity baseline ---")
        list_proba = generate_for_popularity_evaluation(dataset_path, embeddings_version="svd")
        print("Popularity probabilities loaded.")
    
    elif eval_type in ["inputfeatures"]:
        print("--- Loading KMeans model for inputfeatures evaluation ---")
        with open(os.path.join(master_path, clustering_path, clusters_filename), "rb") as f:
            kmeans = pickle.load(f)
        centers = kmeans.cluster_centers_
        if use_cuda:
            centroid_ = torch.FloatTensor(centers).to(device=cuda)
        else:
            centroid_ = torch.FloatTensor(centers)
        print(f"Loaded KMeans model with {len(centroid_)} centroids for inputfeatures.")
        
        print("--- Loading song probability lists for each cluster ---")
        song_proba_by_segment = []
        for cluster_id in range(nb_clusters):
            path = os.path.join(master_path, f"{clustering_path}_probas_{embeddings_version}", f"list_proba_{cluster_id}.pkl")
            song_proba_by_segment.append(pickle.load(open(path, "rb")))
        print(f"Loaded song probabilities for {len(song_proba_by_segment)} clusters for inputfeatures.")
    
    # ---------------------------
    # 3. Compute Evaluation Metrics (Precision, Recall, NDCG) in Batches
    # ---------------------------
    print("--- Starting evaluation loop over test samples ---")
    total_samples = len(total_test_dataset)
    batch_size = 32
    num_batches = int(np.ceil(total_samples / batch_size))
    print(f"Processing test set in {num_batches} batches (batch size = {batch_size}).")
    
    current_ndcg = {k_val: [] for k_val in k_val_list}
    current_recalls = {k_val: [] for k_val in k_val_list}
    current_precisions = {k_val: [] for k_val in k_val_list}
    
    start_total = time.time()
    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        batch = total_test_dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        xs_batch = [x for (x, _, _) in batch]
        gt_ids_batch = [y for (_, y, _) in batch]
        gt_lists_batch = [z for (_, _, z) in batch]
        
        batch_features_tensor_test = torch.stack(xs_batch)
        if use_cuda:
            batch_features_tensor_test = batch_features_tensor_test.cuda(device=cuda)
        
        # Forward pass for the batch
        if eval_type in ["full_perso", "semi_perso", "inputfeatures"]:
            predictions_batch = reg(batch_features_tensor_test)
        elif eval_type in ["avgd0stream"]:
            predictions_batch = torch.FloatTensor(xs_batch)
        else:
            raise ValueError("Unknown eval_type.")
        
        # --- For hard-clustering based soft allocation, process each sample in the batch ---
        if eval_type in ["semi_perso", "inputfeatures"]:
            recommended_songs_batch = []
            scores_batch = []
            # For each sample in the batch:
            for i in range(predictions_batch.shape[0]):
                predicted_vector = predictions_batch[i:i+1].detach().cpu().numpy()[0]  # shape (d,)
                # Compute Euclidean distances from predicted_vector to all centroids:
                centers_np = centroid_.detach().cpu().numpy()  # shape (nb_clusters, d)
                distances = np.linalg.norm(centers_np - predicted_vector, axis=1)  # shape (nb_clusters,)
                # Get the 5 closest clusters:
                top_k = 5
                top5_indices = np.argpartition(distances, top_k)[:top_k]
                # Compute soft membership for these clusters using inverse-distance weighting:
                top5_distances = distances[top5_indices]
                inv_d = 1.0 / (top5_distances + 1e-10)
                membership = inv_d / np.sum(inv_d)  # proportions over top5 clusters
                # Allocate songs proportionally with an overshoot margin:
                k_val_max = max(k_val_list)
                margin = config.get("margin", 4)  # margin to overshoot recommendations
                total_alloc = k_val_max + margin
                allocations = np.floor(membership * total_alloc).astype(int)
                remainder = total_alloc - allocations.sum()
                if remainder > 0:
                    frac_parts = (membership * total_alloc) - allocations
                    add_indices = np.argsort(frac_parts)[::-1]
                    for idx in add_indices[:remainder]:
                        allocations[idx] += 1
                # Gather songs from each of the 5 clusters:
                allocated_songs = []
                for j, cluster in enumerate(top5_indices):
                    # Get the sorted song indices for this cluster in descending order:
                    cluster_recs = np.argsort(song_proba_by_segment[cluster])[::-1]
                    allocated_songs.extend(cluster_recs[:allocations[j]])
                # Remove duplicates while preserving order:
                seen = set()
                final_rec = []
                for song in allocated_songs:
                    if song not in seen:
                        seen.add(song)
                        final_rec.append(song)
                # If we don't have enough unique songs, fill with overall ranking from these 5 clusters:
                if len(final_rec) < k_val_max:
                    overall_scores = np.dot(membership, np.array([song_proba_by_segment[c] for c in top5_indices]))
                    overall_sorted = np.argsort(overall_scores)[::-1]
                    for song in overall_sorted:
                        if song not in seen:
                            seen.add(song)
                            final_rec.append(song)
                        if len(final_rec) >= k_val_max:
                            break
                recommended_songs_batch.append(final_rec[:k_val_max])
                # Also keep the overall scores (for ndcg computation later)
                scores_batch.append(overall_scores)
            recommended_songs_batch = np.array(recommended_songs_batch)
            scores_batch = np.array(scores_batch)
        # End hard-clustering branch
        
        # --- Process each sample in the batch for metric computation ---
        for j in range(batch_features_tensor_test.shape[0]):
            gt_ids = gt_ids_batch[j]
            gt_list = gt_lists_batch[j]
            if eval_type in ["full_perso", "avgd0stream"]:
                proba_values = torch.mm(predictions_batch[j:j+1].detach().cpu(), 
                                        song_embeddings_values_.transpose(0, 1))
                recommended_songs = (proba_values.topk(k=k_val_max, dim=1)[1]).tolist()[0]
                scores = np.asarray([proba_values.detach().cpu().numpy()[0].tolist()])
            elif eval_type in ["semi_perso", "inputfeatures"]:
                scores = np.asarray([scores_batch[j]])
                recommended_songs = recommended_songs_batch[j]
            elif eval_type == "popularity":
                scores = np.asarray([list_proba])
                recommended_songs = np.argsort(list_proba)[::-1]
            else:
                raise ValueError("Unknown eval_type")
            
            groundtruth_array = np.array(gt_list, int).reshape(1, -1)
            for k_val in k_val_list:
                intersection = set(gt_ids) & set(recommended_songs[:k_val])
                denom_precision = float(len(gt_ids)) if len(gt_ids) < k_val else float(k_val)
                precision = len(intersection) / denom_precision
                current_precisions[k_val].append(precision)
                recall = len(intersection) / float(len(gt_ids))
                current_recalls[k_val].append(recall)
            for k_val in k_val_list:
                ndcg = ndcg_score(groundtruth_array, np.asarray([scores[0]]), k=k_val)
                current_ndcg[k_val].append(ndcg)
        batch_end_time = time.time()
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"Batch {batch_idx+1}/{num_batches} processed in {batch_end_time - batch_start_time:.4f} seconds.")
    
    total_time = time.time() - start_total
    print(f"Evaluation complete. Processed {total_samples} samples in {total_time:.2f} seconds.")
    
    for k_val in k_val_list:
        avg_ndcg = sum(current_ndcg[k_val]) / float(len(current_ndcg[k_val]))
        avg_recall = sum(current_recalls[k_val]) / float(len(current_recalls[k_val]))
        avg_precision = sum(current_precisions[k_val]) / float(len(current_precisions[k_val]))
        print(f"Average ndcg at {k_val} is: {avg_ndcg:.4f}")
        print(f"Average recall at {k_val} is: {avg_recall:.4f}")
        print(f"Average precision at {k_val} is: {avg_precision:.4f}")
    
    # ---------------------------
    # 4. Standard Deviation Estimation
    # ---------------------------
    print("--- Starting standard deviation estimation ---")
    max_loc = total_samples
    nb_iterations_eval_stddev = config["nb_iterations_eval_stddev"]
    nb_sub_iterations_eval_stddev = config["nb_sub_iterations_eval_stddev"]
    batch_size_stddev = int(total_samples / float(nb_sub_iterations_eval_stddev))
    batch_ndcg_list = {k_val: [] for k_val in k_val_list}
    batch_recall_list = {k_val: [] for k_val in k_val_list}
    batch_precision_list = {k_val: [] for k_val in k_val_list}
    
    for iteration in range(nb_iterations_eval_stddev):
        torch.manual_seed(iteration)
        randInd = torch.randperm(max_loc)
        current_position = 0
        for i in range(nb_sub_iterations_eval_stddev):
            ending_position = min(current_position + batch_size_stddev, max_loc)
            for k_val in k_val_list:
                batch_recall = pd.DataFrame(current_recalls[k_val]).values[randInd[current_position:ending_position]]
                batch_recall_mean = sum(batch_recall) / float(len(batch_recall))
                batch_recall_list[k_val].append(batch_recall_mean[0])
                batch_precision = pd.DataFrame(current_precisions[k_val]).values[randInd[current_position:ending_position]]
                batch_precision_mean = sum(batch_precision) / float(len(batch_precision))
                batch_precision_list[k_val].append(batch_precision_mean[0])
                batch_ndcg = pd.DataFrame(current_ndcg[k_val]).values[randInd[current_position:ending_position]]
                batch_ndcg_mean = sum(batch_ndcg) / float(len(batch_ndcg))
                batch_ndcg_list[k_val].append(batch_ndcg_mean[0])
            current_position += batch_size_stddev
    
    print(f"Standard deviation estimation complete. Processed {total_samples} samples.")
    for k_val in batch_ndcg_list.keys():
        std_ndcg = statistics.stdev(batch_ndcg_list[k_val])
        print(f"stddev ndcg at {k_val} is: {std_ndcg:.4f}")
    for k_val in batch_recall_list.keys():
        std_recall = statistics.stdev(batch_recall_list[k_val])
        print(f"stddev recall at {k_val} is: {std_recall:.4f}")
    for k_val in batch_precision_list.keys():
        std_precision = statistics.stdev(batch_precision_list[k_val])
        print(f"stddev precision at {k_val} is: {std_precision:.4f}")
    
    print("=== Evaluation Finished ===")




def generate_for_popularity_evaluation(dataset_path, embeddings_version="svd"):
    print("Loading popularity evaluation data...")
    df = pd.read_parquet(os.path.join(dataset_path, f"user_features_train_{embeddings_version}.parquet"), engine='fastparquet')
    exploded_data = df[["user_index", "d1d30_songs"]].explode('d1d30_songs').set_index('d1d30_songs')
    grouped_data = exploded_data.groupby(['d1d30_songs']).size()
    popularity_df = pd.DataFrame(grouped_data / float(sum(grouped_data)))
    popularity_df.columns = ["proba"]
    list_proba = []
    for song_index in range(config["nb_songs"]):
        if song_index in popularity_df.index:
            list_proba.append(popularity_df.loc[song_index]["proba"])
        else:
            list_proba.append(0)
    print("Popularity evaluation data loaded.")
    return list_proba
