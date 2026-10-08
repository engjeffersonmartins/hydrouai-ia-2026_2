# %%
# =============================
# Seção 1: Importação das Bibliotecas
# =============================
# Esta seção importa todas as bibliotecas necessárias para o carregamento de dados,
# construção do modelo, avaliação e visualização dos resultados.

import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# %%
# =============================
# Seção 2: Definição de Parâmetros
# =============================
# Defina aqui os parâmetros principais do experimento:
# - T: horizonte de previsão (quantos passos à frente o modelo irá prever)

T = 1  # Horizonte de previsão: prever Q_Afluente no tempo t+T

# %%
# ==================================================
# Seção 3: Carregamento dos Dados e Criação da Saída
# ==================================================
# Carregamos o dataset e removemos a coluna de data.
# Criamos a variável-alvo deslocando Q_Afluente em T passos à frente.
# Linhas com valores faltantes são removidas.

df = pd.read_csv(r'C:\Users\niewi\Documents\GitHub\hydrouai-ia-2026_2\02_Aulas\Aula03\series_preenchidas.csv').drop(columns=["data"])
df["target"] = df["Q_Afluente"].shift(-T)
df.dropna(inplace=True)

# %%
# ==================================================
# Seção 4: Preparação das Variáveis de Entrada e Saída
# ==================================================
# Separamos as variáveis de entrada X (todas exceto Q_Afluente e target)
# e a variável de saída y (Q_Afluente no tempo t+T).
# Dividimos os dados em treino e validação de forma cronológica.
# Em seguida, normalizamos as variáveis de entrada com StandardScaler.

X = df.drop(columns=["Q_Afluente", "target"]).values
y = df["target"].values

split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# %%
# ========================================
# Seção 5: Definição do Modelo MLP
# ========================================
# Definimos um modelo de rede neural do tipo MLP (perceptron multicamada),
# com duas camadas ocultas e uma camada de saída que retorna a previsão de Q_Afluente.

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# %%
# ==================================================
# Seção 6: Treinamento do Modelo com Curva de Convergência
# ==================================================
# Realiza o treinamento do modelo com mini-batches.
# As perdas de treino e validação são registradas para cada época
# para análise de convergência posterior.
model = MLP(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 3000
batch_size = 1024
train_losses = []
val_losses = []

for epoch in range(epochs):
    indices = torch.randperm(X_train.size(0))
    X_train = X_train[indices]
    y_train = y_train[indices]

    epoch_train_loss = 0
    model.train()
    for i in range(0, X_train.size(0), batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / (X_train.size(0) // batch_size + 1)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val).item()
        val_losses.append(val_loss)

    print(f"Época {epoch+1}/{epochs} - Perda Treino: {avg_train_loss:.4f} - Perda Validação: {val_loss:.4f}")

# %%
# ============================================
# Seção 7: Plot da Curva de Convergência
# ============================================
# Aqui visualizamos as curvas de perda de treino e validação ao longo das épocas.
# Isso ajuda a identificar overfitting, underfitting ou estabilidade no treinamento.

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Perda de Treino", linewidth=1.5)
plt.plot(val_losses, label="Perda de Validação", linewidth=1.5)
plt.title("Curva de Convergência das Perdas")
plt.xlabel("Época")
plt.ylabel("Erro Quadrático Médio (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# ==================================================
# Seção 8: Avaliação no Conjunto de Validação
# ==================================================
# Avaliamos o modelo usando MAE (erro absoluto médio)
# e NSE (eficiência de Nash-Sutcliffe).
# Também visualizamos as previsões versus os valores observados.

val_pred_np = val_preds.numpy().flatten()
y_val_np = y_val.numpy().flatten()

mae = mean_absolute_error(y_val_np, val_pred_np)
nse = 1 - np.sum((y_val_np - val_pred_np)**2) / np.sum((y_val_np - np.mean(y_val_np))**2)

print(f"\n📊 Métricas de Avaliação (Validação):")
print(f"MAE: {mae:.4f}")
print(f"NSE: {nse:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(y_val_np, label="Observado", linewidth=1.5)
plt.plot(val_pred_np, label="Previsto", linewidth=1.5)
plt.title(f"Previsão de Q_Afluente (t+{T}) - Validação")
plt.xlabel("Índice Temporal")
plt.ylabel("Q_Afluente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# ==================================================
# Seção 9: Previsão para Todo o Conjunto de Dados
# ==================================================
# Geramos previsões para todos os dados disponíveis
# e comparamos com os valores reais para análise geral.

X_full = df.drop(columns=["Q_Afluente", "target"]).values
X_full_scaled = scaler.transform(X_full)
X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_full_pred = model(X_full_tensor).numpy().flatten()

y_full_true = df["target"].values

plt.figure(figsize=(14, 5))
plt.plot(y_full_true, label="Observado", linewidth=1.5)
plt.plot(y_full_pred, label="Previsto", linewidth=1.5)
plt.title(f"Previsão de Q_Afluente (t+{T}) - Conjunto Completo")
plt.xlabel("Índice Temporal")
plt.ylabel("Q_Afluente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
