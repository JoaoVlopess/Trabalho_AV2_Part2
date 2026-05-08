import cv2
import numpy as np
import os


def carregar_dataset_hierarquico(caminho_base, largura=50, altura=50):
    X = []
    y = []
    
    # Busca todos os caminhos para as pastas com os dados de cada pessoa
    pastas_pessoas = sorted([d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))])
    
    for i, nome_pasta in enumerate(pastas_pessoas):
        # Para cada pasta com dados de uma pessoa pega-se inicialmente o caominho para aquela base da pessoa especifica a cada iteração
        caminho_completo_pasta = os.path.join(caminho_base, nome_pasta)
        
        # Pega todas as fotos presentes em cada pasta representando a pessoa especifica de iteração
        fotos = [f for f in os.listdir(caminho_completo_pasta) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for nome_foto in fotos:
            # Pega o caminho de cada foto dentro da pasta em questao (com angulos diferentes do rosto da pessoa)
            caminho_foto = os.path.join(caminho_completo_pasta, nome_foto)
            
            # Para cada imagem carregue-a com tons de cinza (mais leve)
            img = cv2.imread(caminho_foto, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Faz-se a interpolação (olhando a area total do bloco de pixels) da imagem para a altura e largura definida ns parametros
                img_res = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)
                
                # Transforma a matriz (largura x altura) para um unico vetor e normaliza cada elemento por 255 visto que na escala de cinza usamos 0 até 255 (forcando cada elemento ficar entre 0 e 1)
                X.append(img_res.flatten() / 255.0)
                
                # O Label será o índice da pasta (0 a 19)
                y.append(i)
                
    return np.array(X), np.array(y), pastas_pessoas

X, y, nomes = carregar_dataset_hierarquico("data/RecFac")
print(f"Total de imagens: {X.shape[0]}")
print(f"Atributos por imagem (d): {X.shape[1]}")

X_transposto = X.T

N = X.shape[0]
linha_bias = np.ones((1, N))

X_final = np.vstack((linha_bias, X_transposto))

