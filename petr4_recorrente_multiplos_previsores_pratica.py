from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Dropout, LSTM # atualizado: tensorflow==2.0.0-beta1
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # atualizado: tensorflow==2.0.0-beta1

base = pd.read_csv('petr4_treinamento.csv')

# apagar os valores faltantes
base = base.dropna()

# dados que serão usados da base
base_treinamento = base.iloc[:, 1:7].values

# normalizacao dos dados
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

# normalizacao dos dados da previsao
normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(base_treinamento[:,0:1])

previsores = []
preco_real = []
for i in range(90, 1242):
    
    previsores.append(base_treinamento_normalizada[i-90:i, 0:6])
    
    # valor do atributo "Open"
    preco_real.append(base_treinamento_normalizada[i, 0])
    
# transformacao de lista para numpy array
previsores, preco_real = np.array(previsores), np.array(preco_real)

# não é preciso fazer reshape igual o exercicio anterior pois ja está no formato correto

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'sigmoid'))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

# parar de fazer o processamento antes de acordo com algumas condições
# para o treinamento quando uma quantidade monitorada parou de melhorar os resultados
# monitor = funcao q vai ser monitorada
# min_delta = mudança minima para ser considerada uma melhoria
# patiente = numero de epocas q vai seguir sem melhorias no resultado
# verbose = mostrar algumas mensagens na tela
es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)

# reduzir a taxa de aprendizagem quando uma metrica parar de melhorar
# factor = valor que a learn rate será reduzida
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)

# salvar o modelo em cada uma das epocas
# filepath = caminho do arquivo em que sera salvo
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', 
                      save_best_only = True, verbose = 1)

regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32,
              callbacks = [es, rlr, mcp])

base_teste = pd.read_csv('petr4_teste.csv')

preco_real_teste = base_teste.iloc[:, 1:2].values

# criando uma variavel para salvar as 2 bases de dados
frames = [base, base_teste]

# concatenacao do frames (as 2 bases de dados)
base_completa = pd.concat(frames)

# remover a coluna date
base_completa = base_completa.drop('Date', axis = 1)

entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0:6])
X_teste = np.array(X_teste)

previsoes = regressor.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()
    
plt.plot(preco_real_teste, color = 'red', label = 'PreÃ§o real')
plt.plot(previsoes, color = 'blue', label = 'PrevisÃµes')
plt.title('PrevisÃ£o preÃ§o das aÃ§Ãµes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()
