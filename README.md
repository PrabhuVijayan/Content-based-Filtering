# Content-Based Movie Recommender System

This project implements a content-based recommender system using a two-tower neural network architecture. It learns user and item embeddings to predict ratings and generate personalized movie recommendations.

# Project Structure
recommender/
│
├── data/ # Dataset and new user input (CSV)
│
├── tools/ # Utility modules
│ ├── data_loader.py # Load dataset and user data
│ ├── preprocessing.py # Scaling and train/test split
│ └── metrics.py # RMSE, MAE, Precision@K, etc.
│
├── models/
│ └── two_tower.py # Neural network model definition & training
│
├── recommend.py # Inference: recommendations & similarity search
├── evaluation.py # Evaluation (regression + ranking metrics)
├── pipeline.py # End-to-end training pipeline
├── config.py # Hyperparameters and configuration
└── main.py # Entry point (training + evaluation + inference)

# Features
- Two-tower neural network for user–item embedding learning  
- Rating prediction (MSE-based training)  
- Top-K recommendation for:
  - New users (from preference input)
  - Existing users  
- Item similarity using learned embeddings  
- Evaluation metrics:
  - Regression: RMSE, MAE, R²  
  - Ranking: Precision@K, Recall@K, HitRate@K, NDCG@K
 
# Usage
Run the full pipeline:

```bash
python main.py

# Current Performance
RMSE ≈ 0.65
Precision@10 ≈ 0.15
Recall@10 ≈ 0.10
HitRate@10 ≈ 0.72
NDCG@10 ≈ 0.18

# Future Improvements
Model selection / architecture tuning to improve NDCG
Ranking-based loss functions (beyond MSE)
Hyperparameter optimization
Better handling of user preferences (e.g., negative feedback)
Faster similarity search (e.g., approximate nearest neighbors)
Support for batch user recommendations
