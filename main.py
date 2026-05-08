import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.madaline import MadalineMulticlasse
from models.mlp import MLP
from models.perceptron_simples import simples_perceptron
from utils.data_loader import carregar_dataset_hierarquico

def one_hot_encoding(y, num_classes):
    n = len(y)
    
    target_matrix = -np.ones((num_classes, n))
    
    target_matrix[y, np.arange(n)] = 1
    
    return target_matrix


X_faces, y_faces, nomes = carregar_dataset_hierarquico("data/RecFac")
d_completo = one_hot_encoding(y_faces, num_classes=20)


# ------------------------------------------------------------------
d_pessoa0 = d_completo[0,:]
modelo_teste = simples_perceptron(learning_rate=0.01, n_epochs=100)

print("Tentando ensinar o Perceptron a reconhecer a Pessoa 0...")
modelo_teste.fit(X_faces, d_pessoa0, patience=50)


plt.plot(modelo_teste.errors)
plt.title("Erro do Perceptron tentando reconhecer uma face")
plt.xlabel("Época")
plt.ylabel("Soma dos Erros")
plt.show()


# -----------------------------------------------------------------

meu_madaline = MadalineMulticlasse(n_classes=20, learning_rate=1e-5, n_epochs=1000)


meu_madaline.fit(X_faces, d_completo)


y_pred = meu_madaline.predict(X_faces)


# Gráfico de Evolução do Erro (EQM) 
plt.figure(figsize=(10, 6))
plt.plot(meu_madaline.errors, color='blue', label='EQM Total')
plt.title("Desempenho do MADALINE Multiclasse (20 Pessoas)")
plt.xlabel("Época")
plt.ylabel("Erro Quadrático Médio (EQM)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()



#--------------------------------------------------------------------
X = X_faces 

D = d_completo.T 

# Configurações da Rede
n_input = X.shape[1]  
n_hidden = 25         
n_output = 20          
learning_rate = 0.0002  
n_epochs = 1000      

# Instância e Treino
modelo_mlp = MLP(n_input, n_hidden, n_output, learning_rate, n_epochs)

print("Iniciando o treinamento do MLP...")
modelo_mlp.fit(X, D)

# Gráfico de Erro
plt.figure(figsize=(10, 6))
plt.plot(modelo_mlp.errors)
plt.title("Evolução do Erro (MSE) - MLP")
plt.xlabel("Épocas")
plt.ylabel("Erro Quadrático Médio")
plt.yscale('log')
plt.grid(True)
plt.show()


previsoes = modelo_mlp.predict(X)
y_pred_mlp = np.argmax(previsoes, axis=1)

acuracia = np.mean(y_pred_mlp == y_faces)
print(f"Acurácia final no treino: {acuracia * 100:.2f}%")

