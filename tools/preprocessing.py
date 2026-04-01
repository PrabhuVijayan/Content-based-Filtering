# import numpy as np
from config import * #noqa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(item_train, user_train, y_train):
    item_train_unscaled = item_train
    user_train_unscaled = user_train
    y_train_unscaled    = y_train

    scalerItem = StandardScaler()
    scalerItem.fit(item_train)
    item_train = scalerItem.transform(item_train)

    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    user_train = scalerUser.transform(user_train)

    scalerTarget = MinMaxScaler((-1, 1))
    scalerTarget.fit(y_train.reshape(-1, 1))
    y_train = scalerTarget.transform(y_train.reshape(-1, 1))
    #ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))
    
    # print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
    # print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))
    # print(np.allclose(y_train_unscaled.reshape(-1,1), scalerTarget.inverse_transform(y_train)))
    
    return(item_train_unscaled, user_train_unscaled, y_train_unscaled, item_train, user_train, y_train, scalerUser, scalerItem, scalerTarget)

def split_data(item, user, y):
    item_tr, item_te = train_test_split(item, train_size=TRAIN_SPLIT, shuffle=True, random_state=1)
    user_tr, user_te = train_test_split(user, train_size=TRAIN_SPLIT, shuffle=True, random_state=1)
    y_tr, y_te       = train_test_split(y,    train_size=TRAIN_SPLIT, shuffle=True, random_state=1)

    return(item_tr, item_te, user_tr, user_te, y_tr, y_te)