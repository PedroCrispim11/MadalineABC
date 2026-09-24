import tkinter as tk
import numpy as np
import os

# ==========================================
# 1. O CÉREBRO (REDE MADALINE)
# ==========================================
class RedeMadaline:
    def __init__(self):
        self.num_entradas = 64
        self.num_classes = 3 # Limitado a 3 (A, B, C) para este teste inicial
        self.alpha = 0.01
        
        self.pesos = np.zeros((self.num_classes, self.num_entradas))
        self.bias = np.zeros(self.num_classes)
        self.alfabeto = "ABC"

    def prever(self, vetor_entrada):
        x = np.array(vetor_entrada)
        saidas = np.dot(self.pesos, x) + self.bias
        indice_vencedor = np.argmax(saidas)
        return self.alfabeto[indice_vencedor]

    def treinar(self, dados_treino, alvos_treino, epocas=100):
        print("Iniciando o treinamento da rede...")
        for epoca in range(epocas):
            erro_total = 0
            
            for x, alvo_idx in zip(dados_treino, alvos_treino):
                x = np.array(x)
                
                # Passa por cada neurônio
                for i in range(self.num_classes):
                    # O alvo (t) é 1 se for a letra do neurônio, senão -1
                    t = 1 if i == alvo_idx else -1
                    
                    # Calcula o sinal de saída
                    v = np.dot(self.pesos[i], x) + self.bias[i]
                    erro = t - v
                    
                    # Atualiza os pesos se houver erro (Regra Delta)
                    if erro != 0:
                        self.pesos[i] += self.alpha * erro * x
                        self.bias[i] += self.alpha * erro
                        erro_total += abs(erro)
            
            # Se a rede decorou os padrões sem errar nada, o treino para
            if erro_total == 0:
                print(f"Rede treinada com sucesso na época {epoca}!")
                break
        print("Pronto para uso.\n")

# ==========================================
# 2. CARREGAMENTO DOS DADOS VIA TXT
# ==========================================
def carregar_dados_do_txt(nome_arquivo):
    dados_treino = []
    alvos_treino = []
    alfabeto_detectado = ""
    
    # Verifica se o arquivo existe
    if not os.path.exists(nome_arquivo):
        print(f"Erro: O arquivo {nome_arquivo} não foi encontrado!")
        return [], [], ""

    with open(nome_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()

    i = 0
    while i < len(linhas):
        linha_atual = linhas[i].strip()
        
        # Se a linha tem exatamente 1 caractere, é a nossa Letra (Rótulo)
        if len(linha_atual) == 1 and linha_atual.isalpha():
            letra = linha_atual.upper()
            
            # Adiciona a letra ao alfabeto se for nova
            if letra not in alfabeto_detectado:
                alfabeto_detectado += letra
            
            alvo_idx = alfabeto_detectado.index(letra)
            
            # Lê as próximas 8 linhas que compõem o desenho
            vetor = []
            for j in range(1, 9):
                if i + j < len(linhas):
                    linha_desenho = linhas[i + j].strip()
                    # Converte 'X' para 1 e '.' para -1
                    for char in linha_desenho:
                        vetor.append(1 if char == 'X' else -1)
            
            dados_treino.append(vetor)
            alvos_treino.append(alvo_idx)
            
            # Pula para a próxima letra (1 linha da letra + 8 do desenho)
            i += 9 
        else:
            i += 1 # Pula linhas em branco
            
    return dados_treino, alvos_treino, alfabeto_detectado

# Lê o arquivo txt
dados_para_treinar, alvos_para_treinar, alfabeto = carregar_dados_do_txt("dataset.txt")

# Inicializa a rede de forma dinâmica
rede = RedeMadaline()
rede.num_classes = len(alfabeto) # Ajusta a quantidade de neurônios automaticamente
rede.alfabeto = alfabeto         # Atualiza as letras que a rede conhece

# Recria os pesos e bias com o novo número de classes
rede.pesos = np.zeros((rede.num_classes, rede.num_entradas))
rede.bias = np.zeros(rede.num_classes)

# Treina a rede
if len(dados_para_treinar) > 0:
    rede.treinar(dados_para_treinar, alvos_para_treinar)

# ==========================================
# 3. A INTERFACE GRÁFICA (TELA)
# ==========================================
def ler_matriz_e_prever():
    vetor_entrada = []
    for linha in matriz_variaveis:
        for var in linha:
            vetor_entrada.append(var.get())
            
    # Manda a lista para o "cérebro" adivinhar
    letra_reconhecida = rede.prever(vetor_entrada)
    
    # Atualiza o texto na tela
    label_resultado.config(text=f"Letra Reconhecida: {letra_reconhecida}")

def limpar_tela():
    for linha in matriz_variaveis:
        for var in linha:
            var.set(-1)
    label_resultado.config(text="Letra Reconhecida: -")

# Configuração da janela
janela = tk.Tk()
janela.title("Reconhecedor Madaline (A, B, C)")
janela.geometry("350x450")

frame_grade = tk.Frame(janela)
frame_grade.pack(pady=20)

matriz_variaveis = []
for i in range(8):
    linha_vars = []
    for j in range(8):
        var = tk.IntVar(value=-1)
        cb = tk.Checkbutton(frame_grade, variable=var, onvalue=1, offvalue=-1)
        cb.grid(row=i, column=j, padx=2, pady=2)
        linha_vars.append(var)
    matriz_variaveis.append(linha_vars)

frame_botoes = tk.Frame(janela)
frame_botoes.pack(pady=10)

btn_ler = tk.Button(frame_botoes, text="Reconhecer Letra", command=ler_matriz_e_prever)
btn_ler.grid(row=0, column=0, padx=10)

btn_limpar = tk.Button(frame_botoes, text="Limpar Grade", command=limpar_tela)
btn_limpar.grid(row=0, column=1, padx=10)

label_resultado = tk.Label(janela, text="Letra Reconhecida: -", font=("Arial", 16, "bold"))
label_resultado.pack(pady=20)

janela.mainloop()