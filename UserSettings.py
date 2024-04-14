from geopy.distance import great_circle
from keras.src.utils import to_categorical
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
from ModelSettings import *


def user_times_arr(username, df):
  timesArr = df['route_start_ts'].unique()
  return timesArr

def get_min_len(df):
  gp = df.groupby('route_start_ts')

  minLen = len(df)
  lastLen = 0
  c = 0
  arrOfLens = list()

  for g in gp.groups.items():
      c += 1
      arrOfLens.append(len(df.loc[g[1]]))

      if len(df.loc[g[1]]) < minLen:
          minLen = len(df.loc[g[1]])
      if c == len(gp.groups.items()):
          lastLen = len(df.loc[g[1]])
          dfNew = df.loc[g[1]]
  return minLen


def get_split_size(df, number):
    gp = df.groupby('route_start_ts')
    fraction = number
    splitSize = int(len(gp) * fraction)
    return splitSize, len(gp)


def train_test_split(df, splitSize):
  myArr = pd.DataFrame()
  myXtest = pd.DataFrame()
  gp = df.groupby('route_start_ts')
  lastLen = 0
  c = 0
  Y = []
  Ytest = []
  Xtest_ = []
  number = get_min_len(df)-1

  for g in gp.groups.items():
      c += 1

      if c <= splitSize:
          lastLen = len(df.loc[g[1]])
          dfNew = df.loc[g[1]]

          a = pd.DataFrame(df.loc[g[1]], columns = df.columns).sort_values(by='ts')
          d = pd.DataFrame(df.loc[g[1]], columns = df.columns).head(number).sort_values(by='ts')
          myXtest = pd.concat([d, myXtest], ignore_index=True)

          Xtest_ = d

          a = a.tail(1)
          a = np.array(a[['lat', 'lon']])
          output_vector = list()
          output_vector.append(a[0][0])
          output_vector.append(a[0][1])
          Ytest.append(output_vector)
          Y.append(output_vector)

      else:
          a = pd.DataFrame(df.loc[g[1]], columns = df.columns).sort_values(by='ts')
          d = pd.DataFrame(df.loc[g[1]], columns = df.columns).head(number).sort_values(by='ts')

          myArr = pd.concat([d, myArr], ignore_index=True)
          a = a.tail(1)
          a = np.array(a[['lat', 'lon']])
          output_vector = list()
          output_vector.append(a[0][0])
          output_vector.append(a[0][1])
          Y.append(output_vector)

  return Xtest_, Ytest, Y, myArr, number, c, myXtest


def dict_safe_zones(sz):
  import warnings
  warnings.simplefilter(action='ignore', category=FutureWarning)
  import pandas
  dictSafeZones = pd.DataFrame(columns=['point', 'class'])

  for i in range(len(sz)):
      point = list()
      point.append(sz.iloc[i]['lat'])
      point.append(sz.iloc[i]['lon'])
      # dictSafeZones = dictSafeZones.append(pd.Series({'point':point, 'class': i}), ignore_index=True)
      dictSafeZones.loc[len(dictSafeZones.index)] = [point, i]

  return dictSafeZones


def get_yclass(dictSafeZones, Y):
  yClass = list()
  for j in range(len(dictSafeZones)):
    yClass.append(j)

  for i in range(len(Y)):
      lastPoint = Y[i]
      minDist = 100000

      for j in range(len(dictSafeZones)):
          currentPoint = dictSafeZones.iloc[j]['point']
          distCheck = great_circle(lastPoint, currentPoint).m
          if distCheck < minDist:
              minDist = distCheck
              classNumber = j
              dictPoint = currentPoint
      # print(minDist, dictPoint ,lastPoint, classNumber)
      yClass.append(classNumber)
  return yClass


def yclass_split(yClass, Ytest, dictSafeZones, testSize, splitSize):
  yClass = to_categorical(yClass)

  Ytest = yClass[-testSize:]
  yClass = yClass[len(dictSafeZones) : splitSize + len(dictSafeZones) ]
  return yClass, Ytest



def yclass_recover(real_arr, pred_arr, dictSafeZones):
  predPoint = dictSafeZones.iloc[0]['point']
  realPoint = dictSafeZones.iloc[0]['point']

  pred_array, real_array = list(), list()
  print('Lennnnnnnnnn', len(pred_arr), len(real_arr), len(dictSafeZones))
  for e in range(len(real_arr)// len(dictSafeZones)):
    for i in range(len(dictSafeZones)):
      if real_arr[len(dictSafeZones) * e + i] == 1:
        realPoint = dictSafeZones.iloc[i]['point']
        real_array.array(realPoint)
        print('Real point is ', realPoint)
      if pred_arr[len(dictSafeZones) * e + i] == 1:
        predPoint = dictSafeZones.iloc[i]['point']
        pred_array.append(predPoint)
        print('Predicted point is ', predPoint)
    print('The distance between points is', great_circle(predPoint, realPoint).m)

  return pred_array, real_array



def get_train_data(myArr, number, c, yClass, myXtest, testSize):

  X_lat = np.array(myArr['lat'])
  X_lon = np.array(myArr['lon'])

  X = np.column_stack((X_lat, X_lon))
  # X = np.array(X).reshape(c-testSize, number, 2)
  # X = np.array(X).reshape(c-testSize, number, 2)
  X = np.array(X).reshape(c-testSize, number, 2)


  Xtest_lat = np.array(myXtest['lat']) # np.array([x+1 for x in range(45)])
  Xtest_lon = np.array(myXtest['lon']) # np.array([x+1 for x in range(45)])

  Xtest = np.column_stack((Xtest_lat, Xtest_lon))
  Xtest = np.array(Xtest).reshape(testSize, number, 2)


  yClass = np.array(yClass)

  return X, yClass, Xtest



