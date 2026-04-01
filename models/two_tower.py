import numpy as np
import tensorflow as tf
from config import * #noqa
from utilities import * #noqa
from tensorflow import keras
from tools.metrics import * #noqa

def build_model(num_user_features, num_item_features, num_outputs=NUM_OUTPUTS): #noqa
    tf.random.set_seed(1)
    user_NN = tf.keras.Sequential([
        tf.keras.layers.Dense(DENSE_LAYERS[0], activation='relu'), #noqa
        tf.keras.layers.Dense(DENSE_LAYERS[1], activation='relu'), #noqa
        tf.keras.layers.Dense(num_outputs, activation='linear')
    ])

    item_NN = tf.keras.Sequential([
        tf.keras.layers.Dense(DENSE_LAYERS[0], activation='relu'), #noqa
        tf.keras.layers.Dense(DENSE_LAYERS[1], activation='relu'), #noqa
        tf.keras.layers.Dense(num_outputs, activation='linear') 
    ])

    input_user = tf.keras.layers.Input(shape=(num_user_features,))
    input_item = tf.keras.layers.Input(shape=(num_item_features,))

    vu = user_NN(input_user)
    vu = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)

    vm = item_NN(input_item)
    vm = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)

    output = tf.keras.layers.Dot(axes=1)([vu, vm])

    model = tf.keras.Model([input_user, input_item], output)
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),  #noqa
        loss='mse')
    
    return model, user_NN, item_NN


def train(model, user_train, item_train, y_train, u_s, i_s, epochs=EPOCHS): #noqa
    model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=epochs)
    return model

def evaluate_ranking(model, user_train_unscaled, item_vecs, user_to_genre,
                     scalerUser, scalerItem, scalerTarget, u_s, i_s, k=TOP_K): #noqa

    user_ids = np.unique(user_train_unscaled[:, 0])[:50]  # sample users

    precisions = []
    recalls = []
    hits = []
    ndcg = []
    
    for uid in user_ids:

        user_vecs, y_true = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre) #noqa

        suser = scalerUser.transform(user_vecs)
        sitem = scalerItem.transform(item_vecs)

        y_pred = model.predict([suser[:, u_s:], sitem[:, i_s:]])
        y_pred = scalerTarget.inverse_transform(y_pred)

        p = precision_at_k(y_true, y_pred, k) #noqa
        r = recall_at_k(y_true, y_pred, k)    #noqa
        h = hit_rate_at_k(y_true, y_pred, k)  #noqa
        n = ndcg_at_k(y_true, y_pred, k)      #noqa

        precisions.append(p)
        recalls.append(r)
        hits.append(h)
        ndcg.append(n)

    print(f"\nRanking Metrics @K={k}")
    print(f"Precision@{k}: {np.mean(precisions):.3f}")
    print(f"Recall@{k}   : {np.mean(recalls):.3f}")
    print(f"HitRate@{k}  : {np.mean(hits):.3f}")
    print(f"NDCG@{k}     : {np.mean(ndcg):.3f}")
    
def evaluate_model(model, user_test, item_test, y_test, scalerTarget, u_s, i_s):
    
    y_pred = model.predict([user_test[:, u_s:], item_test[:, i_s:]])
    y_pred = scalerTarget.inverse_transform(y_pred)
    y_true = scalerTarget.inverse_transform(y_test)

    print("Evaluation Metrics:")
    print(f"RMSE: {rmse(y_true, y_pred):.3f}") #noqa
    print(f"MAE : {mae(y_true, y_pred):.3f}") #noqa
    print(f"R2  : {r2_score(y_true, y_pred):.3f}") #noqa
    
    loss = model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)
    print(f"Test Loss: {loss}")