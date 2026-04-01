import tabulate
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from utilities import * #noqa

def recommend_new_user(model, user_vec, item_vecs,
                       scalerUser, scalerItem, scalerTarget,
                       u_s, i_s):

    # replicate user vector
    user_vecs = gen_user_vecs(user_vec, len(item_vecs)) #noqa

    # scale
    suser = scalerUser.transform(user_vecs)
    sitem = scalerItem.transform(item_vecs)

    # predict
    y_pred = model.predict([suser[:, u_s:], sitem[:, i_s:]])
    y_pred = scalerTarget.inverse_transform(y_pred)

    # sort
    idx = np.argsort(-y_pred, axis=0).reshape(-1)
    return y_pred[idx], item_vecs[idx]

def recommend_existing_user(model, uid,
                            user_train_unscaled, item_vecs, user_to_genre,
                            scalerUser, scalerItem, scalerTarget,
                            u_s, i_s):

    user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre) #noqa

    suser = scalerUser.transform(user_vecs)
    sitem = scalerItem.transform(item_vecs)

    y_pred = model.predict([suser[:, u_s:], sitem[:, i_s:]])
    y_pred = scalerTarget.inverse_transform(y_pred)

    idx = np.argsort(-y_pred, axis=0).reshape(-1)

    return y_pred[idx], y_vecs[idx], user_vecs[idx], item_vecs[idx]

def get_item_embeddings(item_NN, item_vecs, scalerItem, i_s):

    input_item = tf.keras.layers.Input(shape=(item_vecs.shape[1]-i_s,))
    vm = item_NN(input_item)
    vm = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)

    model_m = tf.keras.Model(input_item, vm)

    scaled_items = scalerItem.transform(item_vecs)
    embeddings = model_m.predict(scaled_items[:, i_s:])

    return embeddings

def find_similar_movies(embeddings, item_vecs, movie_dict, sq_dist, top_k=50):

    dim = len(embeddings)
    dist = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            dist[i, j] = sq_dist(embeddings[i], embeddings[j])

    m_dist = ma.masked_array(dist, mask=np.identity(dim))

    results = [["movie1", "genres", "movie2", "genres"]]
    for i in range(top_k):
        j = np.argmin(m_dist[i])
        movie1 = int(item_vecs[i, 0])
        movie2 = int(item_vecs[j, 0])

        results.append([
            movie_dict[movie1]['title'], movie_dict[movie1]['genres'],
            movie_dict[movie2]['title'], movie_dict[movie2]['genres']
            ])
    table = tabulate.tabulate(results, tablefmt='html', headers="firstrow")
    table = table.replace('<th>', '<th style="max-width:100px; word-wrap:break-word;">')
    return table