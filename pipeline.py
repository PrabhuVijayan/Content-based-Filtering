from config import * #noqa
from tools.data_loader import load_data
from tools.preprocessing import scale_features, split_data
from models.two_tower import build_model, train

def run_training_pipeline():

    # Load
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

    # Feature config
    num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
    num_item_features = item_train.shape[1] - 1  # remove movie id at train time
    u_s = U_START  # start of columns to use in training, user #noqa
    i_s = I_START  # start of columns to use in training, items #noqa

    # Preprocess
    item_train_unscaled, user_train_unscaled, y_train_unscaled, item_train, user_train, y_train, scalerUser, scalerItem, scalerTarget = scale_features(item_train, user_train, y_train)

    # Split
    item_train, item_test, user_train, user_test, y_train, y_test = split_data(item_train, user_train, y_train)

    # Model
    model, user_NN, item_NN = build_model(num_user_features, num_item_features)

    # Train
    model = train(model, user_train, item_train, y_train, u_s, i_s)

    return {
        "model": model,
        "user_NN": user_NN,
        "item_NN": item_NN,
        "scalerUser": scalerUser,
        "scalerItem": scalerItem,
        "scalerTarget": scalerTarget,
        "item_vecs": item_vecs,
        "movie_dict": movie_dict,
        "user_to_genre": user_to_genre,
        "user_train_unscaled": user_train_unscaled,
        "user_test": user_test,
        "item_test": item_test,
        "y_test": y_test,
        "u_s": u_s,
        "i_s": i_s
    }