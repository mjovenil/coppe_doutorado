# projeto2_analise.py
# Script de análise e visualização dos resultados — DA vs ES vs Híbrido
# Suporta NC=4 e NC=8 simultaneamente em todos os gráficos.
#
# Dependências:
#   pip install numpy matplotlib pandas seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================================
# 0. Configuração visual global
# =========================================================
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})

COR_DA  = "#1f77b4"   # azul
COR_ES  = "#d62728"   # vermelho
COR_HIB = "#2ca02c"   # verde
COR_REF = "#7f7f7f"   # cinza

# Estilo por NC: NC=4 sólido, NC=8 tracejado
ESTILO = {4: "-", 8: "--"}
MARKER = {4: "o",  8: "s"}

# =========================================================
# 1. Caminhos — ajuste NC aqui se necessário
# =========================================================
HOME       = os.path.expanduser("~")
output_path = os.path.join(HOME, "cpe723_analise")
os.makedirs(output_path, exist_ok=True)

def paths_for_nc(nc):
    return {
        "da":  os.path.join(HOME, f"cpe723_da_nc{nc}",       "grid_search_da_results.csv"),
        "es":  os.path.join(HOME, f"cpe723_es_nc{nc}",       "grid_search_es_results.csv"),
        "hib": os.path.join(HOME, f"cpe723_hibrido_nc{nc}",  "grid_search_hibrido_results.csv"),
    }

J_REF = {4: 0.028944, 8: 0.030216}

# =========================================================
# 2. Carrega dados para NC=4 e NC=8
# =========================================================
def load_csv(path, label):
    if not os.path.exists(path):
        print(f"  [AVISO] não encontrado: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  {label}: {len(df)} combinações")
    return df

def load_npy(path):
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return None
    if not os.path.exists(str(path)):
        return None
    return np.load(str(path))

dados = {}
for nc in [4, 8]:
    p = paths_for_nc(nc)
    print(f"\n--- NC={nc} ---")
    dados[nc] = {
        "da":  load_csv(p["da"],  f"DA  NC={nc}"),
        "es":  load_csv(p["es"],  f"ES  NC={nc}"),
        "hib": load_csv(p["hib"], f"Hib NC={nc}"),
    }

# =========================================================
# 3. Helpers
# =========================================================
def melhor(df, col_sort="SR"):
    if df is None or df.empty:
        return None
    return df.sort_values(col_sort, ascending=False).iloc[0]

def melhor_hib_sr100(df):
    """Melhor híbrido com SR=1.0 e menor AES; fallback para maior SR."""
    if df is None or df.empty:
        return None
    sr1 = df[df["SR"] == 1.0]
    if len(sr1) > 0:
        return sr1.sort_values("AES").iloc[0]
    return df.sort_values("SR", ascending=False).iloc[0]

# =========================================================
# 4. Gráfico 1 — Comparativo SR, MBF, Tempo (barras agrupadas)
# =========================================================
def grafico_comparativo_barras():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Comparativo de Desempenho — Melhor Configuração por Algoritmo",
                 fontweight="bold", fontsize=13)

    ncs       = [4, 8]
    algs      = ["DA", "ES", "Híbrido"]
    cores_alg = [COR_DA, COR_ES, COR_HIB]
    w         = 0.35
    x         = np.arange(len(algs))
    hatches   = {4: "", 8: "//"}
    labels_nc = {4: "NC=4", 8: "NC=8"}

    for ax_i, (ax, metrica, ylabel, yscale) in enumerate(zip(
        axes,
        ["SR", "MBF", "tempo"],
        ["SR (%)", "MBF", "Tempo médio (s)"],
        ["linear", "linear", "log"]
    )):
        for j, nc in enumerate(ncs):
            d = dados[nc]
            vals = []
            for alg_key in ["da", "es", "hib"]:
                b = melhor_hib_sr100(d[alg_key]) if alg_key == "hib" else melhor(d[alg_key])
                if b is None:
                    vals.append(0)
                    continue
                if metrica == "SR":
                    vals.append(b["SR"] * 100)
                elif metrica == "MBF":
                    vals.append(b["MBF"])
                else:
                    vals.append(b["tempo_medio_s"])

            offset = (j - 0.5) * w
            ax.bar(x + offset, vals, width=w,
                   color=cores_alg, edgecolor="white",
                   linewidth=0.8, hatch=hatches[nc],
                   label=labels_nc[nc] if ax_i == 0 else "_nolegend_",
                   alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(algs)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        if metrica == "SR":
            ax.set_ylim(0, 115)
            ax.axhline(100, color=COR_REF, linestyle="--", linewidth=0.8)
        if metrica == "MBF":
            for nc in ncs:
                ax.axhline(J_REF[nc], color=COR_REF,
                           linestyle=ESTILO[nc], linewidth=0.8,
                           label=f"$J/D_{{ref}}$ NC={nc}")
            ax.legend(fontsize=8)

    from matplotlib.patches import Patch
    legend_nc = [Patch(facecolor="gray", hatch="",   label="NC=4"),
                 Patch(facecolor="gray", hatch="//", label="NC=8")]
    axes[0].legend(handles=legend_nc, fontsize=9, loc="lower right")

    plt.tight_layout()
    out = os.path.join(output_path, "comparativo_barras.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 5. Gráfico 2 — Boxplot SR x Algoritmo (NC=4 e NC=8)
# =========================================================
def grafico_boxplot_sr():
    rows = []
    for nc in [4, 8]:
        for alg_key, alg_label in [("da","DA"), ("es","ES"), ("hib","Híbrido")]:
            df = dados[nc][alg_key]
            if df is None:
                continue
            for sr in df["SR"].dropna():
                rows.append({"Algoritmo": alg_label, "NC": f"NC={nc}", "SR (%)": sr*100})

    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_plot, x="Algoritmo", y="SR (%)",
                hue="NC", palette=["#aec7e8","#aec7e8"],
                width=0.6, linewidth=1.2,
                flierprops=dict(marker="o", markersize=4, alpha=0.5), ax=ax)
    sns.stripplot(data=df_plot, x="Algoritmo", y="SR (%)",
                  hue="NC", dodge=True,
                  palette={f"NC={nc}": c for nc, c in [(4,"#1a6fad"),(8,"#d62728")]},
                  size=4, alpha=0.5, jitter=True, ax=ax)

    ax.set_title("Distribuição de SR — Todas as Combinações Testadas",
                 fontweight="bold")
    ax.set_ylabel("Taxa de Sucesso (%)")
    ax.set_xlabel("")
    ax.legend(title="NC", loc="lower right")

    plt.tight_layout()
    out = os.path.join(output_path, "boxplot_sr.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 6. Gráfico 3 — Scatter SR vs Tempo
# =========================================================
def grafico_scatter_sr_tempo():
    """Bubble chart: tamanho do marcador proporcional ao numero de pontos
    sobrepostos na mesma regiao (SR x tempo arredondados)."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 6))
    legend_handles = []

    # Coleta todos os pontos
    todos = []
    for alg_key, label, cor in [("da","DA",COR_DA),
                                  ("es","ES",COR_ES),
                                  ("hib","Híbrido",COR_HIB)]:
        for nc in [4, 8]:
            df = dados[nc][alg_key]
            if df is None: continue
            df_v = df.dropna(subset=["SR","tempo_medio_s"])
            for _, row in df_v.iterrows():
                todos.append({"alg": alg_key, "cor": cor, "label": label,
                               "nc": nc, "sr": row["SR"]*100,
                               "tempo": row["tempo_medio_s"]})

    if not todos:
        return

    df_all = pd.DataFrame(todos)
    df_all["sr_r"]    = df_all["sr"].round(0)
    df_all["tempo_r"] = df_all["tempo"].round(1)

    # Conta sobreposições por grupo
    contagem = (df_all.groupby(["alg","nc","cor","label","sr_r","tempo_r"])
                      .size().reset_index(name="n"))

    n_min, n_max = contagem["n"].min(), contagem["n"].max()
    s_min, s_max = 80, 700

    def tamanho_bubble(n):
        if n_max == n_min:
            return (s_min + s_max) / 2
        return s_min + (n - n_min) / (n_max - n_min) * (s_max - s_min)

    for _, row in contagem.iterrows():
        ax.scatter(row["tempo_r"], row["sr_r"],
                   s=tamanho_bubble(row["n"]),
                   color=row["cor"], alpha=0.75,
                   edgecolors="white", linewidths=0.6,
                   marker=MARKER[row["nc"]])
        if row["n"] > 1:
            ax.annotate(str(int(row["n"])),
                        (row["tempo_r"], row["sr_r"]),
                        fontsize=7, ha="center", va="center",
                        color="white", fontweight="bold")

    # Legenda algoritmo
    for label, cor in [("DA",COR_DA),("ES",COR_ES),("Híbrido",COR_HIB)]:
        legend_handles.append(Line2D([0],[0], marker="o", color="w",
                                     markerfacecolor=cor, markersize=10, label=label))
    # Legenda NC
    for nc, mk in [(4,"o"),(8,"s")]:
        legend_handles.append(Line2D([0],[0], marker=mk, color="gray",
                                     markersize=10, label=f"NC={nc}", linestyle="None"))
    # Legenda tamanho de referência
    for n_ref, lbl in [(1,"1 comb."),(3,"3 comb."),(max(n_max,6),f"{n_max}+ comb.")]:
        n_val = min(n_ref, n_max)
        legend_handles.append(Line2D([0],[0], marker="o", color="w",
                                     markerfacecolor="gray", alpha=0.7,
                                     markersize=np.sqrt(tamanho_bubble(n_val))/2,
                                     label=lbl))

    ax.set_xlabel("Tempo médio por execução (s)")
    ax.set_ylabel("Taxa de Sucesso (%)")
    ax.set_title("Trade-off SR x Tempo — NC=4 e NC=8\n"
                 "(tamanho proporcional ao número de combinações sobrepostas)",
                 fontweight="bold")
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    out = os.path.join(output_path, "scatter_sr_tempo.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 7. Gráfico 4 — Heatmap SR do ES (Nind x Nfilhos)
# =========================================================
def grafico_heatmap_es():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, nc in zip(axes, [4, 8]):
        df = dados[nc]["es"]
        if df is None:
            continue
        df_v = df.dropna(subset=["SR","Nind","Nfilhos"])
        if df_v.empty:
            continue
        pivot = df_v.pivot_table(values="SR", index="Nind",
                                  columns="Nfilhos", aggfunc="max") * 100
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu",
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "SR (%)"},
                    ax=ax, vmin=0, vmax=100)
        ax.set_title(f"Heatmap SR (%) — ES  NC={nc}", fontweight="bold")
        ax.set_xlabel("$N_{{filhos}}$")
        ax.set_ylabel("$N_{{ind}}$")
    plt.tight_layout()
    out = os.path.join(output_path, "heatmap_sr_es.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 8. Gráfico 5 — Heatmap SR do Híbrido (T_da x alpha_da)
# =========================================================
def grafico_heatmap_hibrido():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, nc in zip(axes, [4, 8]):
        df = dados[nc]["hib"]
        if df is None or len(df) < 2:
            ax.set_title(f"Híbrido NC={nc} — sem dados")
            continue
        pivot = df.pivot_table(values="SR", index="T_da",
                                columns="alpha_da", aggfunc="max") * 100
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu",
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "SR (%)"},
                    ax=ax, vmin=0, vmax=100)
        ax.set_title(f"Heatmap SR (%) — Híbrido  NC={nc}", fontweight="bold")
        ax.set_xlabel(r"$\alpha_{DA}$")
        ax.set_ylabel(r"$T_{DA}$")
    plt.tight_layout()
    out = os.path.join(output_path, "heatmap_sr_hibrido.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 9. Gráfico 6 — Convergência 2x2
#    [0,0] DA por iteração    | [0,1] DA por tempo
#    [1,0] ES+Hib por geração | [1,1] ES+Hib por tempo
# =========================================================
def grafico_convergencia_comparativo():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Convergência da Melhor Configuração — DA vs ES vs Híbrido",
                 fontweight="bold", fontsize=13)

    # [0,0] DA por iteração
    ax = axes[0, 0]
    for nc in [4, 8]:
        df = dados[nc]["da"]
        if df is None: continue
        b = melhor(df)
        if b is None: continue
        hist_d = load_npy(b.get("history_Dbest_file"))
        hist_T = load_npy(b.get("history_T_file"))
        if hist_d is None: continue
        ax.plot(hist_d, color=COR_DA, linestyle=ESTILO[nc],
                linewidth=1.8, label=f"D  NC={nc}")
        if hist_T is not None:
            ax.plot(hist_T, color="steelblue", linestyle=ESTILO[nc],
                    linewidth=1.2, alpha=0.8, label=f"T  NC={nc}")
        ax.axhline(J_REF[nc], color=COR_REF, linestyle=ESTILO[nc],
                   linewidth=0.9, alpha=0.7,
                   label=f"$D_{{ref}}$ NC={nc}={J_REF[nc]:.4f}")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Valor")
    ax.legend(fontsize=8, title="Deterministic Annealing", title_fontsize=9)

    # [0,1] DA por tempo
    ax = axes[0, 1]
    for nc in [4, 8]:
        df = dados[nc]["da"]
        if df is None: continue
        b = melhor(df)
        if b is None: continue
        hist_d = load_npy(b.get("history_Dbest_file"))
        hist_t = load_npy(b.get("history_time_file"))
        hist_T = load_npy(b.get("history_T_file"))
        if hist_d is None or hist_t is None: continue
        t_plot = np.concatenate([[0], hist_t])
        h_plot = np.concatenate([[hist_d[0]], hist_d])
        ax.plot(t_plot, h_plot, color=COR_DA, linestyle=ESTILO[nc],
                linewidth=1.8, label=f"D  NC={nc}")
        if hist_T is not None:
            T_plot = np.concatenate([[hist_T[0]], hist_T])
            ax.plot(t_plot, T_plot, color="steelblue", linestyle=ESTILO[nc],
                    linewidth=1.2, alpha=0.8, label=f"T  NC={nc}")
        ax.axhline(J_REF[nc], color=COR_REF, linestyle=ESTILO[nc],
                   linewidth=0.9, alpha=0.7,
                   label=f"$D_{{ref}}$ NC={nc}={J_REF[nc]:.4f}")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Valor")
    ax.legend(fontsize=8, title="Deterministic Annealing", title_fontsize=9)

    # [1,0] ES e Híbrido por geração
    ax = axes[1, 0]
    for alg_key, label, cor in [("es","ES",COR_ES),("hib","Híbrido",COR_HIB)]:
        for nc in [4, 8]:
            df = dados[nc][alg_key]
            if df is None: continue
            b = melhor_hib_sr100(df) if alg_key == "hib" else melhor(df)
            if b is None: continue
            hist_d = load_npy(b.get("history_Dbest_file"))
            if hist_d is None: continue
            ax.plot(hist_d, color=cor, linestyle=ESTILO[nc],
                    linewidth=1.8, label=f"{label}  NC={nc}")
    for nc in [4, 8]:
        ax.axhline(J_REF[nc], color=COR_REF, linestyle=ESTILO[nc],
                   linewidth=0.9, alpha=0.7,
                   label=f"$D_{{ref}}$ NC={nc}={J_REF[nc]:.4f}")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Custo")
    ax.legend(fontsize=8, title="ES e Híbrido ES+DA", title_fontsize=9)

    # [1,1] ES e Híbrido por tempo
    ax = axes[1, 1]
    for alg_key, label, cor in [("es","ES",COR_ES),("hib","Híbrido",COR_HIB)]:
        for nc in [4, 8]:
            df = dados[nc][alg_key]
            if df is None: continue
            b = melhor_hib_sr100(df) if alg_key == "hib" else melhor(df)
            if b is None: continue
            hist_d = load_npy(b.get("history_Dbest_file"))
            hist_t = load_npy(b.get("history_time_file"))
            if hist_d is None or hist_t is None: continue
            t_plot = np.concatenate([[0], hist_t])
            h_plot = np.concatenate([[hist_d[0]], hist_d])
            ax.plot(t_plot, h_plot, color=cor, linestyle=ESTILO[nc],
                    linewidth=1.8, label=f"{label}  NC={nc}")
    for nc in [4, 8]:
        ax.axhline(J_REF[nc], color=COR_REF, linestyle=ESTILO[nc],
                   linewidth=0.9, alpha=0.7,
                   label=f"$D_{{ref}}$ NC={nc}={J_REF[nc]:.4f}")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Custo")
    ax.legend(fontsize=8, title="ES e Híbrido ES+DA", title_fontsize=9)

    plt.tight_layout()
    out = os.path.join(output_path, "convergencia_comparativa.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 10. Gráfico 7 — Tabela visual NC=4 e NC=8
# =========================================================
def grafico_tabela_resultados():
    linhas = []
    for nc in [4, 8]:
        for alg_key, alg_label, cor in [("da","DA",COR_DA),
                                          ("es","ES",COR_ES),
                                          ("hib","Híbrido",COR_HIB)]:
            df = dados[nc][alg_key]
            if df is None:
                continue
            b = melhor_hib_sr100(df) if alg_key == "hib" else melhor(df)
            if b is None:
                continue

            if alg_key == "da":
                params = f"T0={b['T0']}, alpha={b['alpha']}"
            elif alg_key == "es":
                params = f"Nind={b['Nind']}, Nfil={b['Nfilhos']}"
            else:
                params = f"T_da={b['T_da']}, alpha_da={b['alpha_da']}"

            aes_str = f"{b['AES']:.0f}" if not (isinstance(b['AES'], float) and np.isnan(b['AES'])) else "--"
            tempo_str = (f"{b['tempo_medio_s']*1000:.1f} ms"
                         if b['tempo_medio_s'] < 0.5
                         else f"{b['tempo_medio_s']:.2f} s")

            linhas.append([f"NC={nc}", alg_label, params,
                           f"{b['SR']*100:.0f}%",
                           f"{b['MBF']:.5f}",
                           aes_str, tempo_str])

    cols = ["NC", "Algoritmo", "Parametros", "SR", "MBF", "AES", "Tempo"]
    cores_alg = {"DA": COR_DA+"33", "ES": COR_ES+"33", "Híbrido": COR_HIB+"33"}

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(linhas) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=linhas, colLabels=cols,
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, linha in enumerate(linhas, start=1):
        alg = linha[1]
        c = COR_DA+"33" if alg=="DA" else COR_ES+"33" if alg=="ES" else COR_HIB+"33"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(c)

    ax.set_title("Tabela Comparativa — NC=4 e NC=8",
                 fontweight="bold", fontsize=13, pad=12)
    plt.tight_layout()
    out = os.path.join(output_path, "tabela_resultados.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 11. Execução
# =========================================================
if __name__ == "__main__":
    print(f"\nGraficos serao salvos em: {output_path}\n")
    print("=" * 55)

    print("\n[1/7] Comparativo de barras...")
    grafico_comparativo_barras()

    print("\n[2/7] Boxplot SR...")
    grafico_boxplot_sr()

    print("\n[3/7] Scatter SR vs Tempo...")
    grafico_scatter_sr_tempo()

    print("\n[4/7] Heatmap SR -- ES...")
    grafico_heatmap_es()

    print("\n[5/7] Heatmap SR -- Hibrido...")
    grafico_heatmap_hibrido()

    print("\n[6/7] Curvas de convergencia...")
    grafico_convergencia_comparativo()

    print("\n[7/7] Tabela visual...")
    grafico_tabela_resultados()

    print(f"\nConcluido. Graficos em: {output_path}")