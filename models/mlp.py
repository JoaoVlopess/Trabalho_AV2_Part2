import numpy as np

import numpy as np

class MLP:
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01, n_epochs=100):
        self.eta = learning_rate
        self.epoch = n_epochs
        
        self.w_hidden = np.random.randn(n_input, n_hidden) * 0.01
        self.w_output = np.random.randn(n_hidden, n_output) * 0.01
        
        self.b_hidden = np.zeros((1, n_hidden))
        self.b_output = np.zeros((1, n_output))
        
        self.errors = []

    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        return 1.0 - np.square(x)
    
    def fit(self, X, d):
       for epoch in range(self.epoch):

            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            
            soma_erro_quadratico = 0
            
            for i in indices:
                # 1. Seleciona uma única amostra 
                xi = X[i:i+1]
                di = d[i:i+1]

                # Forward 
                u_hidden = np.dot(xi, self.w_hidden) + self.b_hidden
                y_hidden = self._tanh(u_hidden)

                u_output = np.dot(y_hidden, self.w_output) + self.b_output
                y_output = self._tanh(u_output)

                # Cálculo do Erro Local
                error = di - y_output
                soma_erro_quadratico += np.mean(np.square(error))

                # Backpropagation (Gradientes locais)
                d_output = error * self._tanh_derivative(y_output)
                error_hidden = np.dot(d_output, self.w_output.T)
                d_hidden = error_hidden * self._tanh_derivative(y_hidden)

                # Atualização Instantânea (Amostra por Amostra)
                self.w_output += np.dot(y_hidden.T, d_output) * self.eta
                self.b_output += np.sum(d_output, axis=0, keepdims=True) * self.eta

                self.w_hidden += np.dot(xi.T, d_hidden) * self.eta
                self.b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.eta

            # Armazena o erro médio da época
            mse = soma_erro_quadratico / X.shape[0]
            self.errors.append(mse)

            if epoch % 100 == 0:
                print(f"Época {epoch} - MSE: {mse}")
        
    def predict(self, X):
        u_hidden = np.dot(X, self.w_hidden) + self.b_hidden
        y_hidden = self._tanh(u_hidden)

        u_output = np.dot(y_hidden, self.w_output) + self.b_output
        y_output = self._tanh(u_output)
            
        return y_output

