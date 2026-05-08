import numpy as np
from models.adaline import adaline # Importando sua classe Adaline

class MadalineMulticlasse:
    def __init__(self, n_classes=20, learning_rate=1e-5, n_epochs=20):
        self.n_classes = n_classes  # Aqui está o nome que você usou no init
        self.n_epochs = n_epochs
        self.eta = learning_rate
        # Criamos a equipe de especialistas
        self.equipe_adalines = [adaline(learning_rate=learning_rate) for _ in range(n_classes)]
        self.errors = []

    def fit(self, X, D):
        N, p = X.shape
        X_bias = np.insert(X, 0, -1, axis=1)
        
        # Inicializa os pesos de cada objeto adaline na lista
        for ada in self.equipe_adalines:
            ada.w = np.random.randn(X_bias.shape[1]) * 0.01

        for epoch in range(self.n_epochs):
            erro_quadratico_medio_total = 0
            
            for j in range(N):
                # Saída linear u de cada especialista para a foto j
                u_outputs = np.array([X_bias[j] @ ada.w for ada in self.equipe_adalines])
                
                # O gabarito One-Hot para esta foto (coluna j)
                d_esperado = D[:, j]
                
                # Erro linear (Desejado - Saída Linear)
                erros_especificos = d_esperado - u_outputs
                
                # Treina cada Adaline com seu respectivo erro
                for idx, ada in enumerate(self.equipe_adalines):
                    if erros_especificos[idx] != 0:
                        ada.w = ada.w + self.eta * erros_especificos[idx] * X_bias[j]
                
                erro_quadratico_medio_total += np.sum(erros_especificos**2)
            
            # EQM Médio: (Soma dos Erros Quadráticos) / (Amostras * Classes)
            self.errors.append(erro_quadratico_medio_total / (N * self.n_classes))
            print(f"Época {epoch+1}/{self.n_epochs} - EQM: {self.errors[-1]:.6f}")

    def predict(self, X):
        X_bias = np.insert(X, 0, -1, axis=1)
        predicoes_finais = []
        
        for amostra in X_bias:
            # Vê qual Adaline "gritou" mais alto (maior potencial u)
            potenciais = np.array([amostra @ ada.w for ada in self.equipe_adalines])
            vencedor = np.argmax(potenciais)
            predicoes_finais.append(vencedor)
            
        return np.array(predicoes_finais)