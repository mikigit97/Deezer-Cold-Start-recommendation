# Deezer-Cold-Start-recommendation
In the original article of Deezer "A Semi-Personalized System for User Cold Start Recommendation on Music Streaming Apps" Deezer tried to tackle the cold start user problem. We suggest an improvement to their method that yields better results. 

Below is a sample README text you can copy and paste into your repository’s README file:

---

# Semi-Personalized Cold Start Recommendation System

## Overview

Music streaming services face a significant challenge when new users register with little or no interaction data—the cold start problem. Poor initial recommendations can adversely affect user experience and retention. The original paper, *"A Semi-Personalized System for User Cold Start Recommendation on Music Streaming Apps"*, addresses this issue by integrating cold users into an existing latent space using clustering and a deep neural network. 

## The original method

The original method, deployed at scale by Deezer, addresses the user cold start problem through a semi-personalized recommendation system that combines a deep neural network with user segmentation. It builds on two latent collaborative filtering models (UT-ALS and TT-SVD) trained on warm users to learn an embedding space of musical preferences. For cold users, demographic data and same-day interaction signals (e.g., streams, searches, onboarding choices) are aggregated into a fixed-size input vector and passed through a feedforward neural network trained to predict embedding vectors in the latent space. Instead of directly recommending nearest-neighbor tracks, the method assigns cold users to precomputed warm user clusters and recommends popular items within the nearest segment, thereby improving robustness to sparse interaction data. The system is fully integrated into production and supports real-time inference using ONNX models served via a Kubernetes-deployed Golang web service.

## Our Improvement

In this repository, we extend and improve upon the original approach by leveraging a hard clustering technique (using KMeans) combined with a proximity-based soft allocation strategy. Our method involves:

- **Proximity-Based Soft Allocation for Cold Users:**  
  When a new cold user registers, we predict their embedding using a pre-trained regression model. We then determine the five closest clusters by computing the Euclidean distance between the predicted embedding and the cluster centroids. For these five clusters, we compute soft membership percentages using inverse-distance weighting so that the memberships sum to 100%.

- **Proportional Song Allocation and Deduplication:**  
  Based on the soft memberships, we allocate recommendation slots proportionally among the five clusters. For example, if a user’s membership is 30% in a particular cluster, about 30% of the recommendations will be drawn from that cluster’s song probability list. To ensure diversity, we overshoot the target (e.g., allocate 54 slots for a target of 50) and then remove duplicates. If duplicates are removed and the list falls short, we fill in with songs from an overall ranking computed from the top clusters.

This approach improves upon the original baseline by ensuring that recommendations reflect the multi-faceted nature of user preferences, while also providing a clear interpretation of which clusters contribute to the final recommendation list.

## Repository Structure

- **`main.py`** – Main script to run the evaluation and generate recommendations.
- **`clustering.py`** – Code for clustering warm user embeddings and generating per-cluster song probability lists.
- **`evaluation_fuzzy_only_on_cold.py`** – Evaluation script implementing our hard clustering-based soft allocation strategy.
- **`model.py`** – Contains the definition of the regression model used for predicting user embeddings.
- **`options.py`** – Configuration file with hyperparameters and settings.

## How to Use

1. **Clone the Repository:**  
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies:**  
   Set up the required Python environment using `pip install -r requirements.txt` or by activating the provided conda environment.

3. **Run Clustering:**  
   Execute the clustering script to generate the KMeans model and precompute the song probability lists.

4. **Evaluate Recommendations:**  
   Run the evaluation script to generate recommendations for cold users and compute evaluation metrics (Precision, Recall, NDCG).

## Conclusion

Our enhanced method combines the strengths of hard clustering and a proximity-based soft allocation mechanism. By proportionally allocating recommendation slots from the top 5 clusters (with an overshoot margin to avoid duplicate recommendations), we deliver a robust and interpretable solution to the cold start problem in music recommendation. Experimental results indicate that this method significantly improves performance compared to traditional baselines.

---

Feel free to adjust any details to match your specific implementation or experimental setup.
