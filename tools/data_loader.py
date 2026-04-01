import csv
import pickle
from numpy import genfromtxt
from collections import defaultdict

def load_data():
    item_train = genfromtxt('./data/content_item_train.csv', delimiter=',')
    user_train = genfromtxt('./data/content_user_train.csv', delimiter=',')
    y_train    = genfromtxt('./data/content_y_train.csv', delimiter=',')
    with open('./data/content_item_train_header.txt', newline='') as f:    #csv reader handles quoted strings better
        item_features = list(csv.reader(f))[0]
    with open('./data/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]
    item_vecs = genfromtxt('./data/content_item_vecs.csv', delimiter=',')

    movie_dict = defaultdict(dict)
    count = 0
    with open('./data/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  # skip header
            else:
                count += 1
                movie_id = int(line[0])
                movie_dict[movie_id]["title"] = line[1]
                movie_dict[movie_id]["genres"] = line[2]

    with open('./data/content_user_to_genre.pickle', 'rb') as f:
        user_to_genre = pickle.load(f)

    return(item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre)