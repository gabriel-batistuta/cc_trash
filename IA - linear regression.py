# Made using Chat GPT 2024

'''
A regressão linear é uma técnica de análise de dados que prevê o valor de dados desconhecidos usando outro valor de dados relacionado e conhecido. Ele modela matematicamente a variável desconhecida ou dependente e a variável conhecida ou independente como uma equação linear.
'''

class LinearRegression:
    def __init__(self):
        # Inicializa os coeficientes (pesos) e o intercepto (termo de bias)
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        """
        Ajusta o modelo de regressão linear aos dados de treinamento.

        Parâmetros:
        - X: Lista de listas representando as variáveis independentes (features) do treinamento.
        - y: Lista representando a variável dependente (target) do treinamento.
        - learning_rate: Taxa de aprendizado para o algoritmo de gradiente descendente.
        - epochs: Número de iterações para o ajuste dos coeficientes.
        """
        n_samples, n_features = len(X), len(X[0])
        
        # Inicializa os coeficientes com zeros
        self.coefficients = [0.0] * n_features
        self.intercept = 0.0  # Inicializa o intercepto com zero

        # Gradiente Descendente para ajustar os coeficientes e o intercepto
        for _ in range(epochs):
            y_pred = [self._predict(xi) for xi in X]  # Calcula as previsões para o conjunto de treinamento
            
            # Inicializa os gradientes dos coeficientes e do intercepto
            d_coefficients = [0.0] * n_features
            d_intercept = 0.0
            
            # Atualiza os gradientes para cada amostra
            for i in range(n_samples):
                error = y_pred[i] - y[i]  # Calcula o erro de predição para a amostra i
                
                # Atualiza os gradientes dos coeficientes com base no erro
                for j in range(n_features):
                    d_coefficients[j] += error * X[i][j]
                
                # Atualiza o gradiente do intercepto
                d_intercept += error

            # Atualiza os coeficientes e o intercepto usando os gradientes
            for j in range(n_features):
                self.coefficients[j] -= (learning_rate / n_samples) * d_coefficients[j]
            self.intercept -= (learning_rate / n_samples) * d_intercept

    def _predict(self, xi):
        """
        Calcula a previsão para uma única amostra xi.

        Parâmetro:
        - xi: Lista representando as features de uma única amostra.

        Retorna:
        - Valor predito (y) para a amostra xi.
        """
        # Calcula a soma ponderada dos coeficientes e das features, adicionando o intercepto
        return sum(coef * xij for coef, xij in zip(self.coefficients, xi)) + self.intercept

    def predict(self, X):
        """
        Faz previsões para um conjunto de dados X.

        Parâmetro:
        - X: Lista de listas representando as variáveis independentes (features) das amostras.

        Retorna:
        - Lista de valores preditos para cada amostra em X.
        """
        # Retorna as previsões para todas as amostras no conjunto de dados X
        return [self._predict(xi) for xi in X]

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de dados de treinamento (X) e alvo (y)
    X_train = [[1, 2], [2, 3], [4, 5], [3, 2]]
    y_train = [3, 6, 10, 7]

    # Instancia o modelo de regressão linear
    model = LinearRegression()
    
    # Treina o modelo nos dados de treinamento
    model.fit(X_train, y_train, learning_rate=0.01, epochs=1000)

    # Dados de teste para prever os valores
    X_test = [[5, 6], [2, 1]]
    
    # Faz previsões para os dados de teste
    predictions = model.predict(X_test)
    
    # Exibe as previsões
    print("Previsões:", predictions)
