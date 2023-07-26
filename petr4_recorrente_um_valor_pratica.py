from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
# para gerar gráficos
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')

# remover valores nulos
base = base.dropna()

# a previsao será apenas do valor "Open"

# separacao da base
# base.iloc[: (tudo), 1:2 (da coluna 1 até a coluna 2)]
base_treinamento = base.iloc[:, 1:2].values

# normalizacao para conseguir colocar os valores numa escala de 0 a 1
# para processar com mais velocidade os dados e o treinamento
normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

# armazenar os 90 valores das 90 datas
previsores = []
preco_real = []

for i in range(90, 1242):
    # appen = adicionar dados na lista
    # 1º linha - 2º coluna
    # 90 porque quero pegar valores no intervalo de 90 dados
    # : = para ir até o i
    # 0 = valor da coluna, já que só tem uma coluna vai ser o 0 mesmo
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    
    preco_real.append(base_treinamento_normalizada[i, 0])
    
# convertendo para o formato de numpy
previsores, preco_real = np.array(previsores), np.array(preco_real)

# previsores irá sofrer a transformacao
# previsores.shape[0] tem o valor de 1152 que é a quantidade de registros
# previsores.shape[1] vai ser o intervalo de tempo que é igual 90
# 1 de apenas 1 indicador, coluna "Open"
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1]
                                     , 1))

# rede neural

regressor = Sequential()

# camada LSTM ao invés de Dense
# units = celulas de memoria
# return_sequences = True pois temos mais camadas a adicionar depois desta
# 1ª camada
regressor.add(LSTM(units= 100, return_sequences = True,
                   input_shape = (previsores.shape[1], 1)))

# zerar 30% das entradas para evitar overfitting
regressor.add(Dropout(0.3))

# 2ª camada
regressor.add(LSTM(units= 50, return_sequences = True))
regressor.add(Dropout(0.3))

# 3ª camada
regressor.add(LSTM(units= 50, return_sequences = True))
regressor.add(Dropout(0.3))

# 4ª camada
regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.3))

# camada de saida
regressor.add(Dense(units= 1, activation= 'linear'))

regressor.compile(optimizer= 'rmsprop', loss= 'mean_squared_error',
                  metrics= ['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs= 200, batch_size= 32)

base_teste = pd.read_csv('petr4_teste.csv')

preco_real_teste = base_teste.iloc[:, 1:2].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)

# len(base_completa = 1264
# len(base_teste) = 22
# fazendo as ultimas acoes sendo buscadas
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values

# -1 para pois não usaremos linhas
entradas = entradas.reshape(-1, 1)

# normalizar os valores das entradas
entradas = normalizador.transform(entradas)

X_teste = []

# 112 é o tamanho das entradas
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
    
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

# preco medio da acao
previsoes.mean()
# preco medio previsto pela rede neural
preco_real_teste.mean()

# grafico    
plt.plot(preco_real_teste, color = 'red', label = 'Preco real')
plt.plot(previsoes, color = 'blue', label = 'Previsoes')
plt.title('Previsao preco das acoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()
