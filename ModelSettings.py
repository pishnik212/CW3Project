import sys

import numpy as np
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from tensorflow.python.keras.saving.saved_model.load import metrics
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
from UserSettings import *
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

class CustomEarlyStopping(Callback):
    def __init__(self, ratio=0.0,
                 patience=0, verbose=0, restore_best_weights=True):
        super(CustomEarlyStopping, self).__init__()

        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater


    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get('val_loss')
        current_train = logs.get('loss')

        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(np.divide(current_train,current_val), self.ratio):
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))


def loss_roc_plot(history, real_arr, pred_arr):
  fpr, tpr, _ = metrics.roc_curve(real_arr, pred_arr)
  auc = metrics.roc_auc_score(real_arr, pred_arr)


  plt.figure(figsize=(10, 5))
  plt.subplot(121)
  plt.title('model train vs validation loss')
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train','validation'], loc='upper right')


  plt.subplot(122)
  plt.title('ROC AUC')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.plot (fpr,tpr, '--b', linewidth=3, label="AUC = "+str(round(auc, 3)))
  plt.plot(fpr,fpr)
  plt.legend(loc=4)

  print(str(auc))
  print(auc, type(auc), np.around(auc, 2))


def my_model(X, yClass, number):
  minRouteLength = len(yClass[0])
  model_lstm = Sequential()
  model_lstm.add(keras.layers.LSTM(100, activation='relu',
                                   input_shape=(number, 2),
                                   recurrent_activation='hard_sigmoid',
                                   go_backwards=True,
                                   unit_forget_bias=True))
  model_lstm.add(Dropout(0.5))

  model_lstm.add(Dense(minRouteLength, activation='softmax'))
  model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model_lstm


def get_history(model_lstm, X, yClass):  # max 81.6
    early_stop = EarlyStopping(monitor='val_loss',  min_delta=0.001, mode='min', patience=3, verbose=1,restore_best_weights=True) # val_loss accuracy
    callbacks = [CustomEarlyStopping(ratio=0.5, patience=1, verbose=1,restore_best_weights=True)]
    return model_lstm.fit(X, yClass, epochs=len(yClass)*2, validation_split=0.2, batch_size=3, callbacks=[callbacks])

def get_history1(model_lstm, X, yClass):  # max 81.6
    early_stop = EarlyStopping(monitor='val_loss',  min_delta=0.001, mode='min', patience=3, verbose=1,restore_best_weights=True) # val_loss accuracy
    callbacks = [CustomEarlyStopping(ratio=0.5, patience=1, verbose=1,restore_best_weights=True)]

    model_lstm.fit(X, yClass, epochs=len(yClass)*2, validation_split=0.2, batch_size=3, callbacks=[callbacks])
    # model_lstm.save('/content/sample_data/my_model_000.keras')
    # model_lstm.load_weights('/content/sample_data/my_model_000.keras')
    return model_lstm




from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
def func(Xtest, number, model_lstm, dictSafeZones, testSize, Ytest):
  X1 = np.array(Xtest).reshape(testSize, number, 2)
  allRes = list()
  allTestOutput = list()
  test_output = model_lstm.predict(X1, verbose=0)
  maxi = -1
  index = -1
  myTestOutput = list()
  for j in range(len(test_output)):
    for i in range(len(test_output[j])):
      if float(test_output[j][i]) > maxi:
        maxi = float(test_output[j][i])
        index = i


  for j in range(len(test_output)):
    myTestOutput = list()
    for i in range(len(test_output[j])):
      if i == index:
        myTestOutput.append(1)
      else:
        myTestOutput.append(0)
    import functools
    res = functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q, myTestOutput, Ytest[j]))
    allRes.append(res)
    allTestOutput.append(myTestOutput)

  arrY = Ytest.flatten()
  arrT = functools.reduce(lambda x,y :x+y ,allTestOutput)
  pred_arr = arrT
  real_arr = arrY
  conf_matrix = confusion_matrix(real_arr, pred_arr)

  acc_res = round(accuracy_score(real_arr, pred_arr) * 100,2)
  f1_res = round(f1_score(real_arr, pred_arr) * 100,2)
  prec_res = round(precision_score(real_arr, pred_arr) * 100,2)
  rec_res = round(recall_score(real_arr, pred_arr) * 100,2)
  # print(conf_matrix)

  print('Accuracy:', acc_res)
  print('F1:', f1_res)
  print('Precision:', prec_res)
  print('Recall:', rec_res) # ,pos_label='positive', average='micro'

  fig1 = plt.figure(figsize=(1,1))
  ax1 = fig1.add_subplot(111)

  ax1 = sns.heatmap(
      conf_matrix, # confusion matrix 2D array
      annot=True, # show numbers in the cells
      fmt='d', # show numbers as integers
      cbar=False, # don't show the color bar
      cmap='flag', # customize color map
      vmax=175 # to get better color contrast
  )

  ax1.set_xlabel("Predicted")
  ax1.set_ylabel("Actual")
  plt.show()

  return acc_res, pred_arr, real_arr, test_output[0], f1_res, prec_res, rec_res



def last_function(username, dataRoutes, dataSafeZones):
    final = pd.DataFrame(
        columns=['acc_res', 'i', 'out', 'predicted', 'real', 'rec_res', 'prec_res', 'f1_res', 'auc_res'])
    # for j in range(len(recognition_arr)):
    #   i = recognition_arr[j]

    sz = dataSafeZones.loc[((dataSafeZones['user_id'] == username))].drop_duplicates()
    df = dataRoutes.loc[((dataRoutes['user_id'] == username))]
    timesArr = user_times_arr(username, df)

    print(username, len(sz))
    if username is not None and len(sz) > 1:
        splitSize, maxSize = get_split_size(df, 0.8)
        testSize = maxSize - splitSize

    Xtest, Ytest, Y, myArr, number, c, myXtest = train_test_split(df, testSize)[0], train_test_split(df, testSize)[
        1], train_test_split(df, testSize)[2], train_test_split(df, testSize)[3], train_test_split(df, testSize)[4], \
    train_test_split(df, testSize)[5], train_test_split(df, testSize)[6]

    print(number, 'номер')

    yRealPoint = Ytest
    dictSafeZones = dict_safe_zones(sz)
    yClass = get_yclass(dictSafeZones, Y)
    yClass, Ytest = yclass_split(yClass, Ytest, dictSafeZones, testSize, splitSize)

    X, yClass, Xtest = get_train_data(myArr, number, c, yClass, myXtest, testSize)[0], \
    get_train_data(myArr, number, c, yClass, myXtest, testSize)[1], \
    get_train_data(myArr, number, c, yClass, myXtest, testSize)[2]


    model = my_model(X, yClass, number)
    model = get_history1(model, X, yClass)

    output = get_output(Xtest, number, model, dictSafeZones, testSize)
    print(output)
    output = read_output(output)
    print(Ytest)

    predPoint = dictSafeZones.iloc[0]['point']
    realPoint = dictSafeZones.iloc[0]['point']

    real_points, pred_points, dists = list(), list(), list()
    for i in range(len(output)):
        for j in range(len(output[i])):
            if output[i][j] == 1:
                predPoint = dictSafeZones.iloc[j]['point']
                pred_points.append(predPoint)
            if Ytest[i][j] == 1:
                realPoint = dictSafeZones.iloc[j]['point']
                real_points.append(realPoint)
        d = great_circle(predPoint, realPoint).m
        dists.append(d)
        print('The distance between points is', d)

    return real_points, pred_points, dists, output, Xtest

def some_f():
    return  4

def get_output(Xtest, number, model_lstm, dictSafeZones, testSize):
  X1 = np.array(Xtest).reshape(testSize, number, 2)
  allRes = list()
  allTestOutput = list()
  test_output = model_lstm.predict(X1, verbose=0)

  return test_output

def read_output(test_output):
  maxi = -1
  index = -1
  myTestOutput = list()
  allTestOutput = list()
  for j in range(len(test_output)):
    for i in range(len(test_output[j])):
      if float(test_output[j][i]) > maxi:
        maxi = float(test_output[j][i])
        index = i


  for j in range(len(test_output)):
    myTestOutput = list()
    for i in range(len(test_output[j])):
      if i == index:
        myTestOutput.append(1)
      else:
        myTestOutput.append(0)
    import functools
    allTestOutput.append(myTestOutput)

  return allTestOutput

def det_time_arr(dataRoutes, username):
    userData = dataRoutes.loc[dataRoutes['user_id'] == username]
    res = userData.groupby('route_start_ts')
    start_times = userData['route_start_ts'].unique()
    start_times.sort()
    splitSize, maxSize = get_split_size(userData, 0.8)
    last_times = start_times[splitSize:maxSize]
    last_time = start_times[-1]
    train_times = start_times[0:splitSize]
    return start_times, train_times, last_times

