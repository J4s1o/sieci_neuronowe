import pandas as pd
import numpy as np
import math
import pandas_datareader as web
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
plt.style.use("bmh")

from google.colab import drive 
drive.mount('/content/gdrive')
lista_spolek = pd.read_csv('gdrive/My Drive/lista_spolek.csv',header=None)[0].tolist()

def split_sequence(seq,n_steps_in, n_steps_out):
  X,y=[],[]
  for i in range(len(seq)):
    end=i+n_steps_in
    out_end=end + n_steps_out
    if out_end> len(seq):
      break
    # Dzielenie sekwencji na: x = poprzednie ceny i wskaźniki, y = przyszłe ceny
    seq_x,seq_y=seq[i:end,:],seq[end:out_end,0]
    
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)

def visualize_training_results(results):
    """
    Plots the loss and accuracy for the training and testing data
    """
    history = results.history
    plt.figure(figsize=(16,5))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure(figsize=(16,5))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

def layer_maker(n_layers, n_nodes, activation, drop=None, d_rate=.5):
  #Tworzenie określonej liczby ukrytych sieci dla RNN
  for x in range(1,n_layers+1):
    model.add(LSTM(n_nodes, activation=activation, return_sequences=True))
    try:
      if x % drop ==0:
        model.add(Dropout(d_rate))
    except:
        pass
  
def validater(n_per_in, n_per_out,day):
   predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]]) 

   for i in range(n_per_in, len(df)-n_per_in):
     x=df[-i-n_per_in:-i]
     yhat=model.predict(np.array(x).reshape(1,n_per_in,n_features))
     yhat=close_scaler.inverse_transform(yhat)[0]
     yhat2 = yhat[day]
     pred_df=pd.DataFrame(yhat2, 
                               index=pd.date_range(start=x.index[-1], 
                                                   periods=1, 
                                                   freq="B"),
                               columns=[x.columns[0]])
     predictions.update(pred_df)
   return predictions


def val_rsme(df1,df2):
  df=df1.copy()
  df['close2']=df2.Close
  df.dropna(inplace=True)
  df['diff']=df.Close- df.close2
  rms=(df[['diff']]**2).mean()
  return float(np.sqrt(rms))
def obliczWartoscPortfela(portfel, ceny):

  return portfel[0]*ceny + portfel[1]

def kupAkcje(pieniadze, ceny,portfel):

  portfel = [0,0.0]
  iloscAkcji = math.ceil(pieniadze/ceny)-1
  portfel[0] += iloscAkcji
  portfel[1] = pieniadze
  portfel[1] = portfel[1] - iloscAkcji*ceny
  
  return portfel
def SprzedajAkcje(ilosc_akcji,ceny, portfel):
  pom=ilosc_akcji
  portfel[0]=0
  portfel[1]+=pom*ceny
  return portfel
  
a=np.empty((len(lista_spolek),910))
for h in range(0,len(lista_spolek)):
  df = web.DataReader(lista_spolek[h], data_source='yahoo', start='2013-01-01')
  da = df
  df.shape
  print(df)

  df=df.filter(['Close'])
  #Bierze 1000 ostatnich notowań 
  df=df.tail(1000)
  #Skalowanie
  close_scaler=RobustScaler()
  close_scaler.fit(df)
  #Normalizacja
  scaler= RobustScaler()
  df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns, index=df.index)
  # Na ilu dniach mamy uczyć predykcje
  n_per_in  = 90
  # Ile dni predykcji chcemy osiągnąc
  n_per_out = 2
  # Features 
  n_features = df.shape[1]
  # dzielenie danych na odpowiednie sekwencje
  X, y = split_sequence(df.to_numpy(), n_per_in, n_per_out)
  #Tworzenie modelu
  model=Sequential()
  activ="tanh"
  #Warstwa wejściowa
  model.add(LSTM(90,activation=activ,return_sequences=True,input_shape=(n_per_in,n_features)))
  #Ukryta warstwa
  layer_maker(1,30,activ)
  #Końcowa warstwa ukryta
  model.add(LSTM(60,activ))
  #Warstwa wyjściowa
  model.add(Dense(n_per_out))

  model.summary()

  model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
  res=model.fit(X,y,epochs=10,batch_size=128,validation_split=0.1)
  #Przewidywanie ostatnich dni
  lista=[]
  lista2=[]
  lista3=[]
  for i in range (90,1000):
    result=df.head(i)
    #print(result)
    yhat=model.predict(np.array(result.tail(n_per_in)).reshape(1,n_per_in,n_features))
  #print(df.tail(n_per_in))
  #print(np.array(df.tail(n_per_in)).reshape(1,n_per_in,n_features))
    yhat=close_scaler.inverse_transform(yhat)[0]
  #print(yhat)
  #Tworzenie DataFrame z przewidywana ceną
    preds=pd.DataFrame(yhat,index=pd.date_range(start=result.index[-1]+timedelta(days=1),periods=len(yhat),freq="B"),columns=[df.columns[0]])
    pers=n_per_in
  #Transformacja do prawdziwej ceny
    actual=pd.DataFrame(close_scaler.inverse_transform(result.tail(pers)),index=result.Close.tail(pers).index, columns=[df.columns[0]]).append(preds.head(1))
    #print(preds)
    #print(actual)
    #print(actual.iloc[-2:-1].values)
    #print(preds.head(2).values)
    #preds=preds.to_string(index=False)
    lista.append(actual.iloc[-2:-1].values)
    lista2.append(preds.iloc[1].values)
    lista3.append(preds.iloc[2].values)
  rows=len(lista)
  cols=3
  arr=[]
  for i in range(rows):
      col = []
      for j in range(i,i+1):
          col.append(lista[j][0][0])
          col.append(lista2[j][0])
          col.append(lista3[j][0])
          #print(lista2[j][0][0])
      arr.append(col)
  sygnal=[]
  flaga=0
  flaga2=0
  flaga3=0
  for i in range(0,len(arr)):
    if arr[i][0] > arr[i][1] and arr[i][0]>arr[i][2] and flaga==1:
      if flaga3==1:
        sygnal.append(-1)
        flaga=0
        flaga3=0
      elif flaga3==0:
        flaga3=1
        sygnal.append(0)
    elif arr[i][0] < arr[i][1] and arr[i][0]<arr[i][2] and flaga==0:
      if flaga2==1:
        sygnal.append(1)
        flaga=1
        flaga2=0
      elif flaga2==0:
        flaga2=1
        sygnal.append(0)
    else:
      sygnal.append(0)

  for i in range(0,910):
    a[h][i]=sygnal[i]

#Pobieramy ceny i usuwamy z nasaq-100 spółki, które mają za krótki okres notowań
ceny=np.empty((len(lista_spolek)-7,910))
for h in range(0,32):
  df = web.DataReader(lista_spolek[h], data_source='yahoo', start='2013-01-01')
  df=df.filter(['Close'])
  df=df.tail(910)
  df=df.values
  for i in range(0,910):
    ceny[h][i]=df[i]
for h in range(33,40):
  df = web.DataReader(lista_spolek[h], data_source='yahoo', start='2013-01-01')
  df=df.filter(['Close'])
  df=df.tail(910)
  df=df.values
  for i in range(0,910):
    ceny[h-1][i]=df[i]
for h in range(42,62):
  df = web.DataReader(lista_spolek[h], data_source='yahoo', start='2013-01-01')
  df=df.filter(['Close'])
  df=df.tail(910)
  df=df.values
  for i in range(0,910):
    ceny[h-3][i]=df[i]
for h in range(63,76):
  df = web.DataReader(lista_spolek[h], data_source='yahoo', start='2013-01-01')
  df=df.filter(['Close'])
  df=df.tail(910)
  df=df.values
  for i in range(0,910):
    ceny[h-4][i]=df[i]
for h in range(77,78):
  df = web.DataReader(lista_spolek[h], data_source='yahoo', start='2013-01-01')
  df=df.filter(['Close'])
  df=df.tail(910)
  df=df.values
  for i in range(0,910):
    ceny[h-5][i]=df[i]
for h in range(79,101):
  df = web.DataReader(lista_spolek[h], data_source='yahoo', start='2013-01-01')
  df=df.filter(['Close'])
  df=df.tail(910)
  df=df.values
  for i in range(0,910):
    ceny[h-6][i]=df[i]
