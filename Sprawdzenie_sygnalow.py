import pandas as pd
import numpy as np
import math

from google.colab import drive 
drive.mount('/content/gdrive')
price= pd.read_csv("gdrive/My Drive/ceny.csv")
signal=pd.read_csv('gdrive/My Drive/sygnal2.csv')

print(price)
print(signal)

def obliczWartoscPortfela(portfel, ceny):

  suma = 0.0
  for arr in portfel:
    if arr[0] == 'Cash':
      suma += arr[1]
    else:
      suma += arr[1]*ceny[int(arr[0])]

  return suma


def znajdzTicker(portfel, ticker):

  l = -1
  n = len(portfel)
  i = 0
  while i <n:

    if portfel[i][0] == ticker:
      l = i
      i = n
    i=i+1
  
  return l

def kupAkcje(pieniadze, cena,portfel, ticker, wierszCash):

  iloscAkcji = math.ceil(pieniadze/cena)-1
  wiersz= []
  wiersz.append(ticker)
  wiersz.append(iloscAkcji)
  portfel.append(wiersz)
  portfel[wierszCash][1] -= iloscAkcji*cena 

  return portfel

def iloscPieniedzyDoKupna(portfel,wierszCash):

  n = len(portfel)

  k = 11-n

  if k > 0:
    return portfel[wierszCash][1]/k
  else:
    return 0.00
def SprzedajAkcje(cena, portfel, wierszSpolka, wierszCash):
  ilosc_akcji = portfel[wierszSpolka][1]
  portfel[wierszCash][1] += ilosc_akcji*cena
  portfel.remove(portfel[wierszSpolka])
  return portfel

portfel = []

wartoscPortfela = 1000000.00
wykres=[]
ref=[]
wiersz = []

wiersz.append('Cash')
wiersz.append(wartoscPortfela)
portfel.append(wiersz)

wierszCash = znajdzTicker(portfel,'Cash')

for i in range(0,910):
  for j in range(0,100):
    if signal.iloc[j,i] == -1:
      ticker = str(j)
      wierszSprzedaz = znajdzTicker(portfel,ticker)
      if wierszSprzedaz != -1:
        portfel = SprzedajAkcje(price.iloc[j,i],portfel, wierszSprzedaz, wierszCash)

  for j in range(0,100):
    if signal.iloc[j,i] == 1:
      ticker = str(j)
      wierszKupno = znajdzTicker(portfel,ticker)
      if wierszKupno == -1:
        pieniadze = iloscPieniedzyDoKupna(portfel,wierszCash)
        if pieniadze > 0:
          portfel = kupAkcje(pieniadze, price.iloc[j,i],portfel, ticker, wierszCash)
  #print(portfel)
  wykres.append(obliczWartoscPortfela(portfel,price.iloc[:,i]))

import pandas_datareader as web
print(wykres)
df = web.DataReader('FTI', data_source='yahoo', start='2017-09-29',end='2021-05-12')
df.tail(910)
pomoc=df.index
import matplotlib.pyplot as plt
plt.plot(pomoc,wykres)
plt.ylabel('Wartość portfela w milionach USD')
plt.title('Strategia')
plt.show()

#Portfel referencyjny

portfelReferencyjny = []

portfelReferencyjny.append(wiersz)

wierszCash = znajdzTicker(portfelReferencyjny,'Cash')

for j in range(0,100):
    ticker = str(j)
    pieniadze = wartoscPortfela/100
    print(pieniadze)
    portfelReferencyjny = kupAkcje(pieniadze, price.iloc[j,0],portfelReferencyjny, ticker, wierszCash)
    print(portfelReferencyjny)
print("Poziom referencyjny ", obliczWartoscPortfela(portfelReferencyjny,price.iloc[:,909]))