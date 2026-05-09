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

# meu_madaline = MadalineMulticlasse(n_classes=20, learning_rate=1e-5, n_epochs=1000)


# meu_madaline.fit(X_faces, d_completo)


# y_pred = meu_madaline.predict(X_faces)


# # Gráfico de Evolução do Erro (EQM) 
# plt.figure(figsize=(10, 6))
# plt.plot(meu_madaline.errors, color='blue', label='EQM Total')
# plt.title("Desempenho do MADALINE Multiclasse (20 Pessoas)")
# plt.xlabel("Época")
# plt.ylabel("Erro Quadrático Médio (EQM)")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()
# plt.show()



#--------------------------------------------------------------------
X = X_faces 

D = d_completo.T 



# Configurações da Simulação
R = 10
acuracias_mlp = []
n_samples = X_faces.shape[0]
n_treino = int(0.8 * n_samples)  # 80% das amostras 

print(f"Iniciando Simulação de Monte Carlo com {R} rodadas (Manual)...")

for r in range(R):
    # Embaralhar os dados 
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    X_shuffled = X_faces[indices]
    y_shuffled = y_faces[indices]
    
    # Particionamento 80/20 
    X_train = X_shuffled[:n_treino]
    y_train = y_shuffled[:n_treino]
    
    X_test = X_shuffled[n_treino:]
    y_test = y_shuffled[n_treino:]
    
    # Gabarito One-Hot 
    d_train_hot = one_hot_encoding(y_train, num_classes=20).T # Transposto para amostra x classe
    
    #  Reset do Modelo 

    dimensao_entrada = X_train.shape[1]
    modelo_mc = MLP(n_input=dimensao_entrada, n_hidden=25, n_output=20, learning_rate=0.001, n_epochs=500)
    
    # Treinamento 
    modelo_mc.fit(X_train, d_train_hot)
    
    # Predição e Acurácia 
    previsoes_raw = modelo_mc.predict(X_test)
    y_pred = np.argmax(previsoes_raw, axis=1)
    
    acc = np.mean(y_pred == y_test)
    acuracias_mlp.append(acc)
    
    print(f"Rodada {r+1}: Acurácia = {acc*100:.2f}%")

# RESULTADOS FINAIS 
media_acc = np.mean(acuracias_mlp)
desvio_acc = np.std(acuracias_mlp)

print("\n" + "="*35)
print(f"RESULTADO FINAL MONTE CARLO (R={R})")
print(f"Média de Acurácia: {media_acc * 100:.2f}%")
print(f"Desvio Padrão: {desvio_acc * 100:.2f}%")
print("="*35)

