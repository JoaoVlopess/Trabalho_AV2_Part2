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

def criar_matriz_confusao(y_real, y_pred, n_classes=20):
    # Inicializa a matriz com zeros
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Preenche a matriz: linha é o real, coluna é o que o modelo previu
    for real, pred in zip(y_real, y_pred):
        cm[real, pred] += 1
    return cm




X_faces, y_faces, nomes = carregar_dataset_hierarquico("data/RecFac")
d_completo = one_hot_encoding(y_faces, num_classes=20)

R = 10
n_samples = X_faces.shape[0]
n_treino = int(0.8 * n_samples)

# Estruturas para armazenar resultados e objetos das rodadas extremas
resultados_perceptron = {"acc": [], "melhor": None, "pior": None}
resultados_madaline = {"acc": [], "melhor": None, "pior": None}

print(f"Iniciando Monte Carlo para Perceptron e MADALINE (R={R})...")

for r in range(R):
    # Particionamento Aleatório 
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    X_train = X_faces[indices[:n_treino]]
    y_train = y_faces[indices[:n_treino]]
    X_test = X_faces[indices[n_treino:]]
    y_test = y_faces[indices[n_treino:]]
    
    # Validação para o PERCEPTRON SIMPLES (Pessoa 0 vs Resto) 
    d_train_p0 = np.where(y_train == 0, 1, -1)
    d_test_p0 = np.where(y_test == 0, 1, -1)
    
    p_simples = simples_perceptron(learning_rate=0.01, n_epochs=100)
    p_simples.fit(X_train, d_train_p0)
    
    y_pred_p = p_simples.predict(X_test)
    acc_p = np.mean(y_pred_p == d_test_p0)
    resultados_perceptron["acc"].append(acc_p)
    
    # Lógica para salvar melhor/pior Perceptron
    dados_p = {"acc": acc_p, "y_real": d_test_p0, "y_pred": y_pred_p, "errors": p_simples.errors.copy(), "rodada": r+1}
    if resultados_perceptron["melhor"] is None or acc_p > resultados_perceptron["melhor"]["acc"]:
        resultados_perceptron["melhor"] = dados_p
    if resultados_perceptron["pior"] is None or acc_p < resultados_perceptron["pior"]["acc"]:
        resultados_perceptron["pior"] = dados_p

    #  Validação para o MADALINE MULTICLASSE (20 Pessoas) 
    d_train_hot = one_hot_encoding(y_train, num_classes=20) 
    
    m_multi = MadalineMulticlasse(n_classes=20, learning_rate=1e-5, n_epochs=100)
    m_multi.fit(X_train, d_train_hot)
    
    y_pred_m = m_multi.predict(X_test)
    acc_m = np.mean(y_pred_m == y_test)
    resultados_madaline["acc"].append(acc_m)
    
    # Lógica para salvar melhor/pior Madaline
    dados_m = {"acc": acc_m, "y_real": y_test, "y_pred": y_pred_m, "errors": m_multi.errors.copy(), "rodada": r+1}
    if resultados_madaline["melhor"] is None or acc_m > resultados_madaline["melhor"]["acc"]:
        resultados_madaline["melhor"] = dados_m
    if resultados_madaline["pior"] is None or acc_m < resultados_madaline["pior"]["acc"]:
        resultados_madaline["pior"] = dados_m

    print(f"Rodada {r+1}/{R} concluída.")

#  FUNÇÕES DE AUXÍLIO E EXIBIÇÃO 

def exibir_estatisticas(nome, lista_acc):
    media = np.mean(lista_acc)
    desvio = np.std(lista_acc)
    print(f"\n--- Estatísticas {nome} ---")
    print(f"Acurácia Média: {media*100:.2f}%")
    print(f"Desvio Padrão: {desvio*100:.2f}%")

def plotar_extremos(dados, titulo_modelo, multiclasse=True):
    n_classes = 20 if multiclasse else 2
    cm = criar_matriz_confusao(dados['y_real'], dados['y_pred'], n_classes=n_classes)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusão - {titulo_modelo}\n(Rodada {dados['rodada']} - Acc: {dados['acc']*100:.2f}%)")
    
    plt.subplot(1, 2, 2)
    plt.plot(dados['errors'])
    plt.title(f"Curva de Aprendizado - {titulo_modelo}")
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.tight_layout()
    plt.show()

# Chamadas Finais
exibir_estatisticas("PERCEPTRON (Pessoa 0)", resultados_perceptron["acc"])
exibir_estatisticas("MADALINE (Multiclasse)", resultados_madaline["acc"])

# Plotar Melhor e Pior para o Perceptron
plotar_extremos(resultados_perceptron["melhor"], "Perceptron MELHOR", multiclasse=False)
plotar_extremos(resultados_perceptron["pior"], "Perceptron PIOR", multiclasse=False)

# Plotar Melhor e Pior para o Madaline
plotar_extremos(resultados_madaline["melhor"], "Madaline MELHOR", multiclasse=True)
plotar_extremos(resultados_madaline["pior"], "Madaline PIOR", multiclasse=True)
#--------------------------------------------------------------------
X = X_faces 

D = d_completo.T 



# Configurações da Simulação

print(f"Iniciando Simulação de Monte Carlo com {R} rodadas (Manual)...")

melhor_acc = -1
pior_acc = 2.0
dados_melhor = {}
dados_pior = {}

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

    if acc > melhor_acc:
        melhor_acc = acc
        dados_melhor = {
            'y_real': y_test, 
            'y_pred': y_pred, 
            'errors': modelo_mc.errors.copy(),
            'rodada': r + 1
        }

        if acc < pior_acc:
            pior_acc = acc
            dados_pior = {
                'y_real': y_test, 
                'y_pred': y_pred, 
                'errors': modelo_mc.errors.copy(),
                'rodada': r + 1
            }
    
    print(f"Rodada {r+1}: Acurácia = {acc*100:.2f}%")

# RESULTADOS FINAIS 
media_acc = np.mean(acuracias_mlp)
desvio_acc = np.std(acuracias_mlp)

print("\n" + "="*35)
print(f"RESULTADO FINAL MONTE CARLO (R={R})")
print(f"Média de Acurácia: {media_acc * 100:.2f}%")
print(f"Desvio Padrão: {desvio_acc * 100:.2f}%")
print("="*35)

def plotar_resultados_extremos(dados, titulo):
    # Matriz de Confusão
    cm = criar_matriz_confusao(dados['y_real'], dados['y_pred'])
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1 Heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusão - {titulo}\n(Rodada {dados['rodada']})")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    
    # Subplot 2: Curva de Aprendizado
    plt.subplot(1, 2, 2)
    plt.plot(dados['errors'])
    plt.title(f"Curva de Aprendizado - {titulo}")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

# Chama para os dois casos
plotar_resultados_extremos(dados_melhor, "MAIOR ACURÁCIA")
plotar_resultados_extremos(dados_pior, "MENOR ACURÁCIA")




# CONFIGURAÇÕES DA SIMULAÇÃO (R=10)

R = 10
n_samples = X_faces.shape[0]
n_treino = int(0.8 * n_samples)

# Dicionário para armazenar todas as acurácias
acuracias = {
    "Perceptron": [],
    "Madaline": [],
    "MLP": []
}

print(f"Iniciando Simulação de Monte Carlo com {R} rodadas...")
print("Aviso: Com R=100 e 3 modelos, isso pode levar alguns minutos. Pegue um café! ☕")

for r in range(R):
    #  Particionamento Aleatório Justo (Igual para todos na rodada)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    X_train = X_faces[indices[:n_treino]]
    y_train = y_faces[indices[:n_treino]]
    X_test = X_faces[indices[n_treino:]]
    y_test = y_faces[indices[n_treino:]]
    

    # MODELO 1: PERCEPTRON SIMPLES (Pessoa 0 vs Resto)

    d_train_p0 = np.where(y_train == 0, 1, -1)
    d_test_p0 = np.where(y_test == 0, 1, -1)
    
    p_simples = simples_perceptron(learning_rate=0.01, n_epochs=100)
    p_simples.fit(X_train, d_train_p0)
    y_pred_p = p_simples.predict(X_test)
    acuracias["Perceptron"].append(np.mean(y_pred_p == d_test_p0))
    

    # MODELO 2: MADALINE MULTICLASSE 

    d_train_hot_m = one_hot_encoding(y_train, num_classes=20) 
    
    m_multi = MadalineMulticlasse(n_classes=20, learning_rate=1e-5, n_epochs=100)
    m_multi.fit(X_train, d_train_hot_m)
    y_pred_m = m_multi.predict(X_test)
    acuracias["Madaline"].append(np.mean(y_pred_m == y_test))
    

    # MODELO 3: MLP 

    # Usando o Target transposto (amostra x classe) para o  MLP
    d_train_hot_mlp = one_hot_encoding(y_train, num_classes=20).T
    dim_entrada = X_train.shape[1] 
    
    mlp_model = MLP(n_input=dim_entrada, n_hidden=100, n_output=20, learning_rate=0.001, n_epochs=300)
    mlp_model.fit(X_train, d_train_hot_mlp)
    y_pred_mlp = np.argmax(mlp_model.predict(X_test), axis=1)
    acuracias["MLP"].append(np.mean(y_pred_mlp == y_test))
    
    if (r + 1) % 10 == 0:
        print(f"[{r+1}/{R}] rodadas concluídas...")

# CÁLCULO DAS MÉTRICAS E TABELA NO CONSOLE
print("\n" + "="*70)
print(f"{'MODELO':<15} | {'MÉDIA (%)':<10} | {'DESVIO PADRÃO (%)':<18} | {'MÁXIMO (%)':<10} | {'MÍNIMO (%)':<10}")
print("-" * 70)

metricas_finais = {}

for modelo, lista_acc in acuracias.items():
    arr = np.array(lista_acc) * 100 # Multiplica por 100 para virar porcentagem
    
    media = np.mean(arr)
    desvio = np.std(arr)
    maximo = np.max(arr)
    minimo = np.min(arr)
    
    metricas_finais[modelo] = {'media': media, 'std': desvio, 'max': maximo, 'min': minimo}
    
    # Imprime a linha da tabela formatada
    print(f"{modelo:<15} | {media:>9.2f} | {desvio:>17.4f} | {maximo:>10.2f} | {minimo:>10.2f}")

print("="*70)


# GERAÇÃO DO BOXPLOT (GRÁFICO FINAL)
plt.figure(figsize=(10, 6))

# Prepara os dados para o Seaborn (lista de listas)
dados_plot = [np.array(acuracias["Perceptron"])*100, 
              np.array(acuracias["Madaline"])*100, 
              np.array(acuracias["MLP"])*100]

sns.boxplot(data=dados_plot, palette="Set2")
plt.xticks(ticks=[0, 1, 2], labels=["Perceptron (Binário)", "MADALINE (Multiclasse)", "MLP (Multiclasse)"])
plt.title(f"Distribuição das Acurácias após {R} Rodadas (Monte Carlo)", fontsize=14)
plt.ylabel("Acurácia (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6, axis='y')

plt.show()