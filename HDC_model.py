from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ucimlrepo import fetch_ucirepo

# Função para gerar um vetor binário aleatório {-1, 1}
def gerar_vetor_binario(dimension = 10000):
    return np.random.choice([-1, 1], size=dimension)

#Binding: vinculativo;
#Binding → Liga partes específicas para formar estruturas complexas (Ex: cor + forma → maçã vermelha)
# Função de binding: combinação (associação) via multiplicação elemento a elemento (equivalente a XOR binário)
def binding(v1, v2):
    return v1 * v2

#Bundling: agrupamento.
#Bundling → Resume várias instâncias ou ocorrências para formar um "prototipo" ou representação média (Ex: ideia geral do que é uma maçã).
# Função de bundling: combinação por soma e threshold.
# Bundling (combinação) por soma + threshold
def bundling(vetores):
    soma = np.sum(vetores, axis=0)
    return np.where(soma >= 0, 1, -1)

# Similaridade cosseno
def similaridade(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Função de codificação termômetro para variáveis contínuas
def codificacao_termometro(valor, n_niveis):
    limite_superior = n_niveis - 1
    nivel_ativo = int(round(valor * limite_superior))
    vetor = np.full(n_niveis, -1)
    vetor[:nivel_ativo + 1] = 1
    return vetor

# Classe do classificador HDC
class HDCClassificador:
    def __init__(self, d_dimensao=10000, n_niveis=10, modo = 'record'):
        self.DIMENSION = d_dimensao
        self.n_niveis = n_niveis
        self.modo = modo  # 'record' ou 'ngram'
        self.vetores_atributos = {}  # Vetores aleatórios para cada atributo e nível
        self.vetores_posicoes = {}   # Vetores aleatórios para posições dos atributos
        self.prototipos_classes = {}  # Vetores protótipos de cada classe

    def _codificar_exemplo(self, exemplo):
        vetores = []

        if self.modo == 'record':
            for i, valor in enumerate(exemplo):
                if i not in self.vetores_posicoes:
                    self.vetores_posicoes[i] = gerar_vetor_binario(self.DIMENSION)
                vetor_posicao = self.vetores_posicoes[i]

                for nivel in range(self.n_niveis):
                    chave = (i, nivel)
                    if chave not in self.vetores_atributos:
                        self.vetores_atributos[chave] = gerar_vetor_binario(self.DIMENSION)

                vetor_termo = codificacao_termometro(valor, self.n_niveis)
                vetores_nivel = []
                for nivel in range(self.n_niveis):
                    if vetor_termo[nivel] == 1:
                        vetor_nivel = self.vetores_atributos[(i, nivel)]
                        vetor_bind = binding(vetor_nivel, vetor_posicao)
                        vetores_nivel.append(vetor_bind)

                if vetores_nivel:
                    vetor_atributo = bundling(vetores_nivel)
                    vetores.append(vetor_atributo)

        elif self.modo == 'ngram':
            for i in range(len(exemplo) - 1):
                valor1, valor2 = exemplo[i], exemplo[i + 1]

                if i not in self.vetores_posicoes:
                    self.vetores_posicoes[i] = gerar_vetor_binario(self.DIMENSION)
                if (i + 1) not in self.vetores_posicoes:
                    self.vetores_posicoes[i + 1] = gerar_vetor_binario(self.DIMENSION)

                vetor_pos1 = self.vetores_posicoes[i]
                vetor_pos2 = self.vetores_posicoes[i + 1]

                for nivel1 in range(self.n_niveis):
                    chave1 = (i, nivel1)
                    if chave1 not in self.vetores_atributos:
                        self.vetores_atributos[chave1] = gerar_vetor_binario(self.DIMENSION)
                for nivel2 in range(self.n_niveis):
                    chave2 = (i + 1, nivel2)
                    if chave2 not in self.vetores_atributos:
                        self.vetores_atributos[chave2] = gerar_vetor_binario(self.DIMENSION)

                termo1 = codificacao_termometro(valor1, self.n_niveis)
                termo2 = codificacao_termometro(valor2, self.n_niveis)

                vetores_ngram = []
                for n1 in range(self.n_niveis):
                    for n2 in range(self.n_niveis):
                        if termo1[n1] == 1 and termo2[n2] == 1:
                            bind1 = binding(self.vetores_atributos[(i, n1)], vetor_pos1)
                            bind2 = binding(self.vetores_atributos[(i + 1, n2)], vetor_pos2)
                            ngram_bind = binding(bind1, bind2)
                            vetores_ngram.append(ngram_bind)

                if vetores_ngram:
                    vetor_ngram_final = bundling(vetores_ngram)
                    vetores.append(vetor_ngram_final)

        return bundling(vetores)
    
    def treinar(self, X, y):
        prototipos = defaultdict(list)
        for exemplo, classe in zip(X, y):
            vetor = self._codificar_exemplo(exemplo)
            prototipos[classe].append(vetor)

        self.prototipos_classes = {
            classe: bundling(vetores) for classe, vetores in prototipos.items()
        }

    def prever(self, X):
        predicoes = []
        for exemplo in X:
            vetor = self._codificar_exemplo(exemplo)
            melhor_classe = None
            maior_sim = -np.inf
            for classe, prototipo in self.prototipos_classes.items():
                sim = similaridade(vetor, prototipo)
                if sim > maior_sim:
                    maior_sim = sim
                    melhor_classe = classe
            predicoes.append(melhor_classe)
        return predicoes
    
# Função de avaliação completa
def avaliar_modelo(nome, y_true, y_pred, nomes_classes, show_confusion_matrix=True):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n=== {nome} ===")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    if show_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=nomes_classes, yticklabels=nomes_classes)
        plt.title(f'Matriz de Confusão - {nome}')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Previsto')
        plt.show()

def processar_id(identificador_dataset):
    # Busca dataset pelo identificador da UCI
    dataset = fetch_ucirepo(id=identificador_dataset)
    # Seleciona somente colunas numéricas
    
    if dataset.data.targets is None:
        print(f"Dataset '{identificador_dataset}' não tem rótulo. Pulando.")
        return None, None, None, None
    
    matriz_X = dataset.data.features.select_dtypes(include=[np.number]).dropna(axis=1).values
    vetor_y = dataset.data.targets.iloc[:,0].astype('category').cat.codes.values
    #nomes_colunas = dataset.data.features.columns.tolist()
    nomes_classes = dataset.data.targets.iloc[:,0].astype('category').cat.categories.tolist()
    numero_classes = len(np.unique(vetor_y))

    # Divide entre treino e teste
    # Separar treino e teste
    scaler = MinMaxScaler()
    X_normalizado = scaler.fit_transform(matriz_X)
    X_train, X_test, y_train, y_test = train_test_split(X_normalizado, vetor_y, test_size=0.3, random_state=42, stratify=vetor_y)

    # Normalizar os dados entre 0 e 1
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    total_entradas = X_train.shape[1]

    return nomes_classes, numero_classes, X_train, y_train, Counter(y_train), X_test, y_test, Counter(y_test), total_entradas
    
if __name__ == "__main__":
    PRINT_OTHER_OBSERVATIONS = False  # Variável para controlar a impressão de observações adicionais
    DIMENSION = 10000  # Definir a dimensionalidade dos vetores hiperdimensionais com tamanho típico em HDC:10000
        
    print(f"Dimensão dos vetores: {DIMENSION}")
    
    datasets = {
        'iris': 53,
        #'adult': 2,
        #'secondary_mushroom' : 848,
        #'cdc_diabetes_health': 891,
    }
    
    for dataset, id_dataset in datasets.items():
        nomes_classes, numero_classes, X_train, y_train, quantity_y_train, X_test, y_test, quantity_y_test, total_entradas = processar_id(id_dataset)
        n_niveis = len(nomes_classes)**2  # Níveis de codificação termômetro
        
        print(f"""
Nome da classe: {dataset}, total de colunas/features: {total_entradas}
Classes: {nomes_classes}, quantidade de classes: {numero_classes}
Quantidade de treino: {len(y_train)}, quantidade de teste: {len(y_test)}
Níveis de codificação: {n_niveis}""")
    
        # Record-based
        hdc_record = HDCClassificador(d_dimensao=DIMENSION, n_niveis=n_niveis, modo='record')
        print("Iniciando treinamento record...")
        hdc_record.treinar(X_train, y_train)
        print("Treinamento concluído.\nPrevendo...")
        pred_record = hdc_record.prever(X_test)
        print("Previsão concluída.\nAvaliando modelo...")
        avaliar_modelo("HDC - Record-based", y_test, pred_record, nomes_classes)
        print("Avaliação record concluída.\n")
        
        # N-gram based
        hdc_ngram = HDCClassificador(d_dimensao=DIMENSION, n_niveis=n_niveis, modo='ngram')
        print("Iniciando treinamento N-gram...")
        hdc_ngram.treinar(X_train, y_train)
        print("Treinamento concluído.\nPrevendo...")
        pred_ngram = hdc_ngram.prever(X_test)
        print("Previsão concluída.\nAvaliando modelo...")
        avaliar_modelo("HDC - N-gram based", y_test, pred_ngram, nomes_classes)
        print("Avaliação N-gram concluída.\n")