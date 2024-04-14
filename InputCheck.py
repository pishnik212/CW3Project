import matplotlib.pyplot as plt
from geopy.distance import great_circle
import pandas as pd
import numpy as np
# import gdown
from keras.utils import to_categorical
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import SGD, RMSprop
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import tensorflow
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def routes_columns(dataRoutes):
    columns = ['route_start_ts', 'route_end_ts', 'user_id', 'ts', 'lat', 'lon']
    return sorted(columns) == sorted(dataRoutes.columns)

def sz_columns(dataSafeZones):
    columns = ['user_id', 'lat', 'lon', 'radius']
    return sorted(columns) == sorted(dataSafeZones.columns)

def check_number_sz(dataSafeZones, dataRoutes):
    dsz = dataSafeZones.groupby(['user_id']).size().to_frame(name='size').reset_index()
    dsz = dsz.loc[dsz['size'] >= 2, 'user_id']
    usernames = dsz.values.tolist()
    dataSafeZones = dataSafeZones[dataSafeZones.user_id.isin(usernames)]
    dataRoutes = dataRoutes[dataRoutes.user_id.isin(usernames)]
    return dataSafeZones, dataRoutes


def prepocessing(dataSafeZones, dataRoutes):
    dataSafeZones, dataRoutes = check_number_sz(dataSafeZones, dataRoutes)
    return 0




def func_sz(dataSafeZones):
    return dataSafeZones.loc[((dataSafeZones['user_id'] == 92914108))].drop_duplicates()

def user_tables(dataSafeZones, username):
    return dataSafeZones.loc[((dataSafeZones['user_id'] == username))].drop_duplicates()



