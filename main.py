import random
import webbrowser
import numpy as np
import pandas as pd
import tensorflow as tf
from config import * #noqa
from recommend import * #noqa
from tools.metrics import * #noqa
from models.two_tower import * #noqa
from tools.data_loader import * #noqa
from tools.preprocessing import * #noqa
from pipeline import run_training_pipeline
pd.set_option("display.precision", 1)
np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)

# Run pipeline
data = run_training_pipeline()

uvs = UVS  # user genre vector start #noqa
ivs = IVS  # item genre vector start #noqa

# Evaluate
evaluate_model(                #noqa
    data["model"],
    data["user_test"],
    data["item_test"],
    data["y_test"],
    data["scalerTarget"],
    data["u_s"],
    data["i_s"])

evaluate_ranking(      #noqa
    data["model"],
    data["user_train_unscaled"],
    data["item_vecs"],
    data["user_to_genre"],
    data["scalerUser"],
    data["scalerItem"],
    data["scalerTarget"],
    data["u_s"],
    data["i_s"],
    k=TOP_K)   #noqa

# Predictions for New user
user_vec =  pd.read_csv("data/new_user.csv").values
y_pred, items = recommend_new_user(data["model"], user_vec, data["item_vecs"], #noqa
    data["scalerUser"], data["scalerItem"], data["scalerTarget"],
    data["u_s"], data["i_s"])

html4 = print_pred_movies(y_pred, items, data["movie_dict"], maxcount = 10) #noqa
with open("table4.html", "w") as f:
    f.write(html4)
webbrowser.open("table4.html")

# Predictions for an existing user
uid=2
y_pred, y_true, users, items = recommend_existing_user(data["model"], uid, data["user_train_unscaled"], #noqa
    data["item_vecs"], data["user_to_genre"], data["scalerUser"], data["scalerItem"], data["scalerTarget"], data["u_s"], data["i_s"])

#print sorted predictions for movies rated by the user
html5 = print_existing_user(y_pred, y_true.reshape(-1,1), users, items, ivs, uvs, data["movie_dict"], maxcount = 50) #noqa
with open("table5.html", "w") as f:
    f.write(html5)
webbrowser.open("table5.html")

# Finding similar movies
embeddings = get_item_embeddings(data["item_NN"], data["item_vecs"], data["scalerItem"], data["i_s"]) #noqa
similar = find_similar_movies(embeddings, data["item_vecs"], data["movie_dict"], sq_dist) #noqa
with open("table6.html", "w") as f:
    f.write(similar)
webbrowser.open("table6.html")