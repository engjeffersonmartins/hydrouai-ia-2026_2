# %%
# ==============================================================
# MLP COM INCERTEZA EPISTEMICA VIA MONTE CARLO DROPOUT
# ==============================================================
# O arquivo "series_preenchidas.csv" deve estar na mesma pasta
# deste script. Todos os caminhos sao relativos.
#
# Ideia:
# - Durante o treinamento, Dropout regulariza a rede.
# - Durante a previsao, mantemos o Dropout ATIVO e executamos a
#   mesma entrada muitas vezes (Monte Carlo Dropout).
# - A media das simulacoes e a previsao central.
# - A dispersao entre simulacoes representa INCERTEZA EPISTEMICA,
#   isto e, incerteza associada ao proprio modelo/parametros.
#
# Observacao importante:
# esta banda NAO representa toda a incerteza preditiva. Ela nao inclui
# explicitamente a incerteza aleatoria/aleatoria dos dados (aleatoria).

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# %%
# ==============================================================
# 1. PARAMETROS PRINCIPAIS
# ==============================================================

T = 7                      # horizonte de previsao em passos de tempo
TRAIN_RATIO = 0.80         # fracao cronologica usada para treinamento
EPOCHS = 3000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

# Parametros da incerteza epistemica
DROPOUT_RATE = 0.15        # pode testar 0.05, 0.10, 0.15, 0.20, 0.30
N_MC = 300                 # numero de previsoes estocasticas por amostra
INTERVALO = 0.95           # intervalo epistemico mostrado (95%)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_DADOS = BASE_DIR / "series_preenchidas.csv"
ARQUIVO_RESULTADO = BASE_DIR / "previsoes_mlp_incerteza_epistemica.csv"


# %%
# ==============================================================
# 2. CARREGAMENTO DOS DADOS
# ==============================================================

if not ARQUIVO_DADOS.exists():
    raise FileNotFoundError(
        f"Arquivo nao encontrado: {ARQUIVO_DADOS}\n"
        "Coloque 'series_preenchidas.csv' na mesma pasta do script."
    )

df = pd.read_csv(ARQUIVO_DADOS)

if "data" not in df.columns:
    raise ValueError("O arquivo precisa conter a coluna 'data'.")
if "Q_Afluente" not in df.columns:
    raise ValueError("O arquivo precisa conter a coluna 'Q_Afluente'.")

df["data"] = pd.to_datetime(df["data"], errors="coerce")
df = df.sort_values("data").reset_index(drop=True)

# alvo futuro
df["target"] = df["Q_Afluente"].shift(-T)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("=" * 70)
print("DADOS")
print("=" * 70)
print(f"Arquivo: {ARQUIVO_DADOS.name}")
print(f"Registros utilizados: {len(df)}")
print(f"Periodo: {df['data'].min().date()} a {df['data'].max().date()}")


# %%
# ==============================================================
# 3. ENTRADAS E SAIDA
# ==============================================================

COLUNAS_EXCLUIR = ["data", "Q_Afluente", "target"]
feature_names = [c for c in df.columns if c not in COLUNAS_EXCLUIR]

X = df[feature_names].to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.float32)
datas = df["data"].to_numpy()

split_idx = int(len(X) * TRAIN_RATIO)

X_train_np = X[:split_idx]
X_val_np = X[split_idx:]
y_train_np = y[:split_idx]
y_val_np = y[split_idx:]

datas_train = datas[:split_idx]
datas_val = datas[split_idx:]

# Normalizacao ajustada apenas no treino
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_val_scaled = scaler.transform(X_val_np)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

print(f"Variaveis de entrada: {len(feature_names)}")
print(f"Treino: {len(X_train)}")
print(f"Validacao: {len(X_val)}")


# %%
# ==============================================================
# 4. MLP COM DROPOUT
# ==============================================================

class MLPDropout(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.15):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


model = MLPDropout(X_train.shape[1], dropout_rate=DROPOUT_RATE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %%
# ==============================================================
# 5. TREINAMENTO
# ==============================================================

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    indices = torch.randperm(X_train.size(0))
    X_train_epoch = X_train[indices]
    y_train_epoch = y_train[indices]

    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, X_train_epoch.size(0), BATCH_SIZE):
        xb = X_train_epoch[i:i + BATCH_SIZE]
        yb = y_train_epoch[i:i + BATCH_SIZE]

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    train_loss = epoch_loss / n_batches
    train_losses.append(train_loss)

    # Para acompanhar a validacao de forma deterministica,
    # desligamos o dropout neste ponto.
    model.eval()
    with torch.no_grad():
        pred_val_det = model(X_val)
        val_loss = criterion(pred_val_det, y_val).item()
    val_losses.append(val_loss)

    if epoch == 0 or (epoch + 1) % 100 == 0:
        print(
            f"Epoca {epoch + 1:4d}/{EPOCHS} | "
            f"Treino: {train_loss:.4f} | Validacao: {val_loss:.4f}"
        )


# %%
# ==============================================================
# 6. CURVA DE CONVERGENCIA
# ==============================================================

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Treino", linewidth=1.3)
plt.plot(val_losses, label="Validacao", linewidth=1.3)
plt.xlabel("Epoca")
plt.ylabel("MSE")
plt.title("Curva de convergencia")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# %%
# ==============================================================
# 7. FUNCAO MONTE CARLO DROPOUT
# ==============================================================

def mc_dropout_predict(model, X_tensor, n_mc=300, intervalo=0.95):
    """
    Executa n_mc previsoes mantendo Dropout ativo.

    Retorna:
    - matriz com todas as simulacoes [n_mc, n_amostras]
    - media das previsoes
    - desvio-padrao epistemico
    - limite inferior do intervalo por quantis
    - limite superior do intervalo por quantis
    """

    # IMPORTANTE: model.train() mantem Dropout ativo durante inferencia.
    model.train()

    simulacoes = []

    with torch.no_grad():
        for _ in range(n_mc):
            pred = model(X_tensor).squeeze(1).cpu().numpy()
            simulacoes.append(pred)

    simulacoes = np.asarray(simulacoes)

    media = np.mean(simulacoes, axis=0)
    desvio = np.std(simulacoes, axis=0, ddof=1)

    alpha = 1.0 - intervalo
    inferior = np.quantile(simulacoes, alpha / 2.0, axis=0)
    superior = np.quantile(simulacoes, 1.0 - alpha / 2.0, axis=0)

    return simulacoes, media, desvio, inferior, superior


# %%
# ==============================================================
# 8. INCERTEZA EPISTEMICA NA VALIDACAO
# ==============================================================

(
    val_mc,
    val_media,
    val_std,
    val_inf,
    val_sup,
) = mc_dropout_predict(
    model,
    X_val,
    n_mc=N_MC,
    intervalo=INTERVALO,
)

# Metricas da previsao central
mae = mean_absolute_error(y_val_np, val_media)
rmse = np.sqrt(mean_squared_error(y_val_np, val_media))

nse_den = np.sum((y_val_np - np.mean(y_val_np)) ** 2)
nse = 1.0 - np.sum((y_val_np - val_media) ** 2) / nse_den

# Metricas relacionadas a incerteza
coberto = (y_val_np >= val_inf) & (y_val_np <= val_sup)
cobertura = 100.0 * np.mean(coberto)
largura = val_sup - val_inf
largura_media = np.mean(largura)
std_media = np.mean(val_std)

# Quanto a incerteza acompanha o erro real?
erro_abs = np.abs(y_val_np - val_media)
if np.std(val_std) > 0 and np.std(erro_abs) > 0:
    correlacao_erro_incerteza = np.corrcoef(erro_abs, val_std)[0, 1]
else:
    correlacao_erro_incerteza = np.nan

print("\n" + "=" * 70)
print("RESULTADOS - VALIDACAO")
print("=" * 70)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"NSE: {nse:.4f}")
print()
print("INCERTEZA EPISTEMICA")
print(f"Monte Carlo passes: {N_MC}")
print(f"Dropout rate: {DROPOUT_RATE:.2f}")
print(f"Intervalo nominal: {INTERVALO * 100:.1f}%")
print(f"Desvio-padrao epistemico medio: {std_media:.4f}")
print(f"Largura media do intervalo: {largura_media:.4f}")
print(f"Cobertura empirica do observado: {cobertura:.2f}%")
print(f"Correlacao |erro| x incerteza: {correlacao_erro_incerteza:.4f}")
print()
print(
    "Observacao: a cobertura acima NAO deve necessariamente ser igual ao "
    f"intervalo nominal de {INTERVALO*100:.0f}%, pois a banda representa apenas "
    "incerteza epistemica, nao a incerteza preditiva total."
)


# %%
# ==============================================================
# 9. GRAFICO PRINCIPAL: PREVISAO + BANDA EPISTEMICA
# ==============================================================

plt.figure(figsize=(14, 6))
plt.plot(datas_val, y_val_np, label="Observado", linewidth=1.3)
plt.plot(datas_val, val_media, label="Previsao media", linewidth=1.3)
plt.fill_between(
    datas_val,
    val_inf,
    val_sup,
    alpha=0.25,
    label=f"Intervalo epistemico {INTERVALO*100:.0f}%",
)
plt.xlabel("Data")
plt.ylabel("Q_Afluente")
plt.title(f"MLP com incerteza epistemica - horizonte t+{T}")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# %%
# ==============================================================
# 10. EVOLUCAO DA INCERTEZA EPISTEMICA
# ==============================================================

plt.figure(figsize=(14, 4))
plt.plot(datas_val, val_std, linewidth=1.2)
plt.xlabel("Data")
plt.ylabel("Desvio-padrao epistemico")
plt.title("Incerteza epistemica do modelo ao longo do tempo")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# %%
# ==============================================================
# 11. ERRO ABSOLUTO x INCERTEZA EPISTEMICA
# ==============================================================

plt.figure(figsize=(7, 6))
plt.scatter(val_std, erro_abs, alpha=0.35)
plt.xlabel("Desvio-padrao epistemico")
plt.ylabel("Erro absoluto")
plt.title(
    "Erro absoluto x incerteza epistemica\n"
    f"Correlacao = {correlacao_erro_incerteza:.3f}"
)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# %%
# ==============================================================
# 12. PREVISAO EPISTEMICA PARA TODO O CONJUNTO
# ==============================================================

X_full_scaled = scaler.transform(X)
X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)

(
    full_mc,
    full_media,
    full_std,
    full_inf,
    full_sup,
) = mc_dropout_predict(
    model,
    X_full_tensor,
    n_mc=N_MC,
    intervalo=INTERVALO,
)


# %%
# ==============================================================
# 13. SALVAMENTO DOS RESULTADOS
# ==============================================================

resultado = pd.DataFrame({
    "data": df["data"],
    "Q_Afluente_observado_t_mais_T": y,
    "previsao_media": full_media,
    "incerteza_epistemica_std": full_std,
    "limite_inferior_epistemico": full_inf,
    "limite_superior_epistemico": full_sup,
    "largura_intervalo_epistemico": full_sup - full_inf,
})

# Adiciona informacao treino/validacao para facilitar analises posteriores
resultado["conjunto"] = "treino"
resultado.loc[split_idx:, "conjunto"] = "validacao"

resultado.to_csv(ARQUIVO_RESULTADO, index=False)

print("\n" + "=" * 70)
print("ARQUIVO GERADO")
print("=" * 70)
print(ARQUIVO_RESULTADO.name)
print()
print("Colunas principais:")
print("- previsao_media")
print("- incerteza_epistemica_std")
print("- limite_inferior_epistemico")
print("- limite_superior_epistemico")
print("- largura_intervalo_epistemico")


# %%
# ==============================================================
# 14. RESUMO DOS PONTOS DE MAIOR INCERTEZA NA VALIDACAO
# ==============================================================

resumo_val = pd.DataFrame({
    "data": pd.to_datetime(datas_val),
    "observado": y_val_np,
    "previsao": val_media,
    "erro_absoluto": erro_abs,
    "std_epistemico": val_std,
    "limite_inferior": val_inf,
    "limite_superior": val_sup,
})

resumo_val = resumo_val.sort_values("std_epistemico", ascending=False)

print("\n10 instantes de MAIOR incerteza epistemica na validacao:")
print(resumo_val.head(10).to_string(index=False))
