# Reproducibility
SEED = 1

U_START = 3
I_START = 1
UVS = 3
IVS = 3

# Model
NUM_OUTPUTS = 32
DENSE_LAYERS = [256, 128]

# Training
EPOCHS = 30
LEARNING_RATE = 0.01
TRAIN_SPLIT = 0.8

# Recommendation
TOP_K = 10
SIMILAR_TOP_K = 50
RELEVANCE_THRESHOLD = 3.5  # or 4.0

# Paths
NEW_USER_PATH = "data/new_user.csv"