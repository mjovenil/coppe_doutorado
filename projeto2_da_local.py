# Deterministic Annealing (DA)
# Versão local — Linux / VS Code
#
# Dependências:
#   pip install numpy matplotlib pandas

import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import os
import datetime
import time
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# 0. Diretório de saída local
# =========================================================
# local_path é definido dinamicamente no __main__ com base em NC_number
local_path = None  # será atualizado em __main__
# csv_path_da e best_record_path_da são definidos em __main__
csv_path_da = None
best_record_path_da = None
print("Resultados DA serão salvos em:")
print(csv_path_da)
print(best_record_path_da)

# =========================================================
# 1. Geração de dados sintéticos em R^3
# =========================================================
def generate_data_r3(P=100, NC=8, sigma=0.1, seed=1):
    np.random.seed(seed)
    cluster_centers = np.random.normal(0, 1, (3, NC))
    data_vectors = []
    for k in range(NC):
        centro = cluster_centers[:, k:k+1]
        cluster = sigma * np.random.normal(0, 1, (3, P)) + np.tile(centro, (1, P))
        data_vectors.append(cluster)
    data_vectors = np.concatenate(data_vectors, axis=1)
    return data_vectors, cluster_centers

# =========================================================
# 2. Função custo "hard" — mesma do ES
# =========================================================
def J_hard(X, Y):
    diff  = X[:, :, None] - Y[:, None, :]
    dist2 = np.sum(diff**2, axis=0)
    return np.mean(np.min(dist2, axis=0))

# =========================================================
# 3. Uma execução do DA (versão vetorizada)
#
# Parâmetros:
#   T0      — temperatura inicial
#   alpha   — fator de redução da temperatura
#   epsilon — perturbação após convergência em cada temperatura
#   delta   — critério de convergência local (variação relativa de J)
#   I       — número máximo de iterações
#   Tmin    — temperatura mínima (critério de parada)
#   tol     — tolerância para considerar sucesso (|D - D_ref| <= tol)
#   D_ref   — custo de referência (centros verdadeiros)
# =========================================================
def run_da(
    data_vectors,
    NC,
    T0=5.0,
    alpha=0.85,
    epsilon=1e-6,
    delta=1e-3,
    I=200,
    Tmin=0.1,
    tol=1e-2,
    D_ref=None,
    init_seed=0
):
    start_time = time.perf_counter()
    np.random.seed(init_seed)

    X    = data_vectors
    M, N = X.shape
    K    = NC

    Y      = np.random.normal(0, 1, (M, K))
    Y_best = Y.copy()
    T      = T0
    D_best = np.inf
    isuc   = 0
    ncalls = 0       # avaliações da função custo
    ncalls_suc = 0   # avaliações até o sucesso

    history_J      = np.zeros(I)
    history_D      = np.zeros(I)
    history_T      = np.zeros(I)
    history_Dbest  = np.zeros(I)
    history_time   = np.zeros(I)

    for i in range(I):

        # Calcula distâncias (K, N): d[k,n] = ||x_n - y_k||^2
        diff = X[:, :, None] - Y[:, None, :]      # (M, N, K)
        d    = np.sum(diff**2, axis=0).T           # (K, N)

        p    = np.exp(-d / T)
        Zx   = np.sum(p, axis=0)                  # (N,)
        p   /= Zx                                 # normaliza: p(k|x_n)

        # Atualiza centróides: y_k = sum_n p(k|x_n)*x_n / sum_n p(k|x_n)
        weights = np.sum(p, axis=1)               # (K,)
        Y       = (X @ p.T) / weights             # (M, K)

        # Métricas
        # 1 chamada por iteração = 1 passagem pela partição + centróide
        # (conforme orientação do professor)
        ncalls += 1
        J_val   = -T / N * np.sum(np.log(Zx))
        D_val   = np.mean(np.sum(p * d, axis=0))

        history_J[i]     = J_val
        history_D[i]     = D_val
        history_T[i]     = T
        history_time[i]  = time.perf_counter() - start_time

        if D_val < D_best:
            D_best = D_val
            Y_best = Y.copy()

        history_Dbest[i] = D_best

        # Verifica sucesso
        if isuc == 0 and D_ref is not None:
            if abs(D_best - D_ref) <= tol:
                isuc       = 1
                ncalls_suc = ncalls

        # Critério de convergência local → reduz temperatura
        if i > 0:
            rel_change = abs(history_J[i] - history_J[i-1]) / max(abs(history_J[i-1]), 1e-12)
            if rel_change < delta:
                T *= alpha
                Y += epsilon * np.random.normal(0, 1, Y.shape)

        # Critério de parada
        if T < Tmin:
            i += 1
            break

    elapsed = time.perf_counter() - start_time

    return (
        D_best, Y_best,
        history_J[:i], history_D[:i],
        history_T[:i], history_Dbest[:i],
        history_time[:i],
        isuc, ncalls_suc
    )

# =========================================================
# 4. Loop de 100 execuções — calcula SR, MBF, AES
#    (mesmo padrão do ES)
# =========================================================
def run_100_execucoes_da(
    data_vectors,
    NC,
    D_ref,
    T0, alpha, epsilon, delta, I, Tmin,
    tol=1e-2,
    Nexec=100,
    base_seed=0
):
    """
    Executa o DA Nexec vezes com seeds independentes e calcula:
      SR  — taxa de sucesso (% execuções com |D - D_ref| <= tol)
      MBF — média do melhor custo encontrado em cada execução
      AES — número médio de avaliações até o sucesso (só nas bem-sucedidas)
    """
    D_runs    = []
    isuc_runs = []
    aes_runs  = []
    best_hist_D      = None
    best_hist_Dbest  = None
    best_hist_time   = None
    best_hist_J      = None
    best_hist_T      = None
    best_D = np.inf

    for i in range(Nexec):
        seed = base_seed + i
        (D_best, Y_best,
         hist_J, hist_D, hist_T, hist_Dbest,
         hist_time, isuc, ncalls_suc) = run_da(
            data_vectors=data_vectors, NC=NC,
            T0=T0, alpha=alpha, epsilon=epsilon,
            delta=delta, I=I, Tmin=Tmin,
            tol=tol, D_ref=D_ref,
            init_seed=seed
        )
        D_runs.append(D_best)
        isuc_runs.append(isuc)
        if isuc:
            aes_runs.append(ncalls_suc)
        if D_best < best_D:
            best_D         = D_best
            best_hist_D    = hist_D.copy()
            best_hist_Dbest= hist_Dbest.copy()
            best_hist_time = hist_time.copy()
            best_hist_J    = hist_J.copy()
            best_hist_T    = hist_T.copy()

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{Nexec} execuções concluídas")

    SR  = float(np.mean(isuc_runs))
    MBF = float(np.mean(D_runs))
    AES = float(np.mean(aes_runs)) if aes_runs else np.nan

    print(f"  SR={SR*100:.1f}%  MBF={MBF:.6f}  AES={AES:.0f}")

    return {
        "SR": SR, "MBF": MBF, "AES": AES,
        "D_best":         best_D,
        "best_hist_D":    best_hist_D,
        "best_hist_Dbest":best_hist_Dbest,
        "best_hist_time": best_hist_time,
        "best_hist_J":    best_hist_J,
        "best_hist_T":    best_hist_T,
    }

# =========================================================
# 5. Grid search para DA
# =========================================================
def grid_search_da(
    data_vectors,
    cluster_centers,
    NC,
    T0_values,
    alpha_values,
    epsilon_values,
    delta_values,
    I_values,
    Tmin_values,
    tol=1e-2,
    Nexec=100,
    N_rep=10,
    csv_path=csv_path_da,
    best_record_path=best_record_path_da
):
    D_global_ref = J_hard(cluster_centers, data_vectors)
    print(f"Custo com centros verdadeiros: {D_global_ref:.6f}")

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        key_cols  = ["T0", "alpha", "epsilon", "delta", "I", "Tmin"]
        available = [c for c in key_cols if c in df_existing.columns]
        tested    = set(tuple(row) for row in df_existing[available].values)
        print(f"{len(df_existing)} linhas já existentes no CSV — serão puladas.")
    else:
        df_existing = pd.DataFrame()
        tested      = set()

    combinations = list(itertools.product(
        T0_values, alpha_values, epsilon_values,
        delta_values, I_values, Tmin_values
    ))
    total = len(combinations)
    print(f"Total de combinações possíveis: {total}")

    best_global_value = np.inf
    if os.path.exists(best_record_path):
        old_best = np.load(best_record_path, allow_pickle=True)
        best_global_value = float(old_best["D_best"])
        print(f"Melhor global prévio encontrado: {best_global_value:.6f}")

    best_record  = None
    t_grid_start = time.perf_counter()

    for idx, (T0, alpha, epsilon, delta, I, Tmin) in enumerate(combinations, start=1):

        comb_key = (T0, alpha, epsilon, delta, I, Tmin)
        if comb_key in tested:
            print(f"[{idx}/{total}] Pulando combinação já calculada.")
            continue

        elapsed_grid = time.perf_counter() - t_grid_start
        eta_str = (str(datetime.timedelta(
                        seconds=int(elapsed_grid / (idx-1) * (total-idx+1))))
                   if idx > 1 else "calculando...")

        print(f"\n{'='*50}")
        print(f"Combinação {idx}/{total}  |  ETA: {eta_str}")
        print(f"T0={T0}, alpha={alpha}, epsilon={epsilon}, "
              f"delta={delta}, I={I}, Tmin={Tmin}")

        # 100 execuções para SR / MBF / AES
        metrics = run_100_execucoes_da(
            data_vectors=data_vectors, NC=NC, D_ref=D_global_ref,
            T0=T0, alpha=alpha, epsilon=epsilon,
            delta=delta, I=I, Tmin=Tmin,
            tol=tol, Nexec=Nexec, base_seed=0
        )

        # Medição de tempo (N_rep repetições, seed=0)
        print(f"  -> Medindo tempo ({N_rep} repetições)...")
        t_runs = []
        for _ in range(N_rep):
            t0 = time.perf_counter()
            run_da(
                data_vectors=data_vectors, NC=NC,
                T0=T0, alpha=alpha, epsilon=epsilon,
                delta=delta, I=I, Tmin=Tmin,
                init_seed=0
            )
            t_runs.append(time.perf_counter() - t0)
        tempo_medio = float(np.mean(t_runs))
        tempo_std   = float(np.std(t_runs))
        print(f"  -> Tempo médio: {tempo_medio:.3f} s | Std: {tempo_std:.3f} s")

        # Salva arquivos
        combo_tag = f"T0{T0}_a{alpha}_e{epsilon}_d{delta}_I{I}_Tmin{Tmin}".replace(".", "p")
        history_D_path     = os.path.join(local_path, f"da_history_D_{combo_tag}.npy")
        history_Dbest_path = os.path.join(local_path, f"da_history_Dbest_{combo_tag}.npy")
        history_time_path  = os.path.join(local_path, f"da_history_time_{combo_tag}.npy")

        np.save(history_D_path,     metrics["best_hist_D"])
        np.save(history_Dbest_path, metrics["best_hist_Dbest"])
        np.save(history_time_path,  metrics["best_hist_time"])
        history_J_path = os.path.join(local_path, f"da_history_J_{combo_tag}.npy")
        history_T_path = os.path.join(local_path, f"da_history_T_{combo_tag}.npy")
        np.save(history_J_path,     metrics["best_hist_J"])
        np.save(history_T_path,     metrics["best_hist_T"])

        row = {
            "timestamp":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "T0":                 T0,
            "alpha":              alpha,
            "epsilon":            epsilon,
            "delta":              delta,
            "I":                  I,
            "Tmin":               Tmin,
            "tol":                tol,
            "Nexec":              Nexec,
            "SR":                 metrics["SR"],
            "MBF":                metrics["MBF"],
            "AES":                metrics["AES"],
            "D_best":             metrics["D_best"],
            "D_global_reference": float(D_global_ref),
            "tempo_medio_s":      tempo_medio,
            "tempo_std_s":        tempo_std,
            "history_D_file":     history_D_path,
            "history_Dbest_file": history_Dbest_path,
            "history_time_file":  history_time_path,
            "history_J_file":     history_J_path,
            "history_T_file":     history_T_path,
        }

        df_row = pd.DataFrame([row])
        write_header = not os.path.exists(csv_path)
        df_row.to_csv(csv_path, mode="a", header=write_header, index=False)
        print(f"  -> SR={metrics['SR']*100:.1f}% | MBF={metrics['MBF']:.6f} | "
              f"AES={metrics['AES']:.0f} | salvo no CSV")

        if metrics["D_best"] < best_global_value:
            best_global_value = metrics["D_best"]
            best_record = {
                "T0": T0, "alpha": alpha, "epsilon": epsilon,
                "delta": delta, "I": I, "Tmin": Tmin,
                "D_best":      metrics["D_best"],
                "SR":          metrics["SR"],
                "MBF":         metrics["MBF"],
                "AES":         metrics["AES"],
                "best_hist_D":     metrics["best_hist_D"],
                "best_hist_Dbest": metrics["best_hist_Dbest"],
                "best_hist_time":  metrics["best_hist_time"],
                "best_hist_J":     metrics["best_hist_J"],
                "best_hist_T":     metrics["best_hist_T"],
            }
            np.savez(
                best_record_path,
                T0=T0, alpha=alpha, epsilon=epsilon,
                delta=delta, I=I, Tmin=Tmin,
                D_best=metrics["D_best"],
                SR=metrics["SR"], MBF=metrics["MBF"], AES=metrics["AES"],
                best_hist_D=metrics["best_hist_D"],
                best_hist_Dbest=metrics["best_hist_Dbest"],
                best_hist_time=metrics["best_hist_time"],
                best_hist_J=metrics["best_hist_J"],
                best_hist_T=metrics["best_hist_T"],
            )
            print(f"  -> NOVO melhor global: {best_global_value:.6f} — salvo")

    print(f"\nGrid search concluído. Tempo total: "
          f"{str(datetime.timedelta(seconds=int(time.perf_counter()-t_grid_start)))}")

    df_results = (
        pd.read_csv(csv_path)
        .sort_values(by="SR", ascending=False)
        .reset_index(drop=True)
    )

    if best_record is None and os.path.exists(best_record_path):
        loaded = np.load(best_record_path, allow_pickle=True)
        best_record = {k: loaded[k] for k in loaded.files}

    return df_results, best_record

# =========================================================
# 6. Plot DA
# =========================================================
def plot_best_solution_da(data_vectors, cluster_centers, best_record):

    history_D     = np.asarray(best_record["best_hist_D"])
    history_Dbest = np.asarray(best_record["best_hist_Dbest"])
    history_time  = np.asarray(best_record["best_hist_time"])
    history_J     = np.asarray(best_record["best_hist_J"])
    history_T     = np.asarray(best_record["best_hist_T"])
    D_ref         = J_hard(cluster_centers, data_vectors)

    # Gráfico 1: -J, D e Temperatura por iteração (estilo original) + linha de referência
    plt.figure(figsize=(10, 6))
    plt.plot(-history_J, 'r-', label='-J')
    plt.plot(history_D,  'k-', label='D')
    plt.plot(history_T,  'b-', label='Temperatura')
    plt.axhline(y=D_ref, color='b', linestyle='--', linewidth=1.0,
                label='Custo com centros verdadeiros')
    plt.grid(); plt.xlabel('Iteração'); plt.ylabel('Valor')
    plt.title('Deterministic Annealing')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(local_path, "da_convergencia_original.png"), dpi=150)
    plt.show()

    # Gráfico 2: -J, D e Temperatura por tempo (s) + linha de referência
    plt.figure(figsize=(10, 6))
    plt.plot(history_time, -history_J, 'r-', label='-J')
    plt.plot(history_time,  history_D, 'k-', label='D')
    plt.plot(history_time,  history_T, 'b-', label='Temperatura')
    plt.axhline(y=D_ref, color='b', linestyle='--', linewidth=1.0,
                label='Custo com centros verdadeiros')
    plt.grid(); plt.xlabel('Tempo (s)'); plt.ylabel('Valor')
    plt.title('Deterministic Annealing — por tempo')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(local_path, "da_convergencia_original_tempo.png"), dpi=150)
    plt.show()

    # Gráfico 3: D e D_best por iteração
    plt.figure(figsize=(10, 6))
    plt.plot(history_D,     'r-', label='D (distorção média)')
    plt.plot(history_Dbest, 'k-', label='Melhor D acumulado')
    plt.axhline(y=D_ref, color='b', linestyle='--', linewidth=1.0,
                label='Custo com centros verdadeiros')
    plt.grid(); plt.xlabel('Iteração'); plt.ylabel('Custo')
    plt.title('Deterministic Annealing — convergência por iteração')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(local_path, "da_convergencia_iteracao.png"), dpi=150)
    plt.show()

    # Gráfico 4: D e D_best por tempo (s)
    plt.figure(figsize=(10, 6))
    plt.plot(history_time, history_D,     'r-', label='D (distorção média)')
    plt.plot(history_time, history_Dbest, 'k-', label='Melhor D acumulado')
    plt.axhline(y=D_ref, color='b', linestyle='--', linewidth=1.0,
                label='Custo com centros verdadeiros')
    plt.grid(); plt.xlabel('Tempo (s)'); plt.ylabel('Custo')
    plt.title('Deterministic Annealing — convergência por tempo')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(local_path, "da_convergencia_tempo.png"), dpi=150)
    plt.show()

# =========================================================
# 7. Medição de tempo de execução
# =========================================================
def medir_tempo_da(data_vectors, NC, best_record, N_rep=10):
    tempos = []
    T0      = float(best_record["T0"])
    alpha   = float(best_record["alpha"])
    epsilon = float(best_record["epsilon"])
    delta   = float(best_record["delta"])
    I       = int(best_record["I"])
    Tmin    = float(best_record["Tmin"])

    print(f"\nMedindo tempo de execução ({N_rep} repetições)...")
    for rep in range(N_rep):
        t0 = time.perf_counter()
        run_da(
            data_vectors=data_vectors, NC=NC,
            T0=T0, alpha=alpha, epsilon=epsilon,
            delta=delta, I=I, Tmin=Tmin,
            init_seed=0
        )
        tempos.append(time.perf_counter() - t0)
        print(f"  rep {rep+1:2d}/{N_rep}: {tempos[-1]:.3f} s")

    media = float(np.mean(tempos))
    std   = float(np.std(tempos))
    print(f"\nTempo médio  : {media:.3f} s")
    print(f"Desvio padrão: {std:.3f} s")
    return media, std

# =========================================================
# 8. Execução principal
# =========================================================
if __name__ == "__main__":

    NC_number = 8   # ← mude aqui para 4, 8, 16, etc.

    # Diretório de saída automático baseado no NC_number
    local_path = os.path.expanduser(f"~/cpe723_da_nc{NC_number}")
    os.makedirs(local_path, exist_ok=True)
    csv_path_da = os.path.join(local_path, "grid_search_da_results.csv")
    best_record_path_da = os.path.join(local_path, "best_record_da.npz")
    print(f"Resultados salvos em: {local_path}")
    data_vectors, cluster_centers = generate_data_r3(P=100, NC=NC_number, sigma=0.1, seed=1)
    D_ref = J_hard(cluster_centers, data_vectors)
    print(f"Custo com centros verdadeiros: {D_ref:.6f}")

    N_rep_timing = 10
    Nexec        = 100

    df_results, best_record = grid_search_da(
        data_vectors=data_vectors,
        cluster_centers=cluster_centers,
        NC=NC_number,
        T0_values      = [5.0, 10.0],
        alpha_values   = [0.85, 0.90, 0.95],
        epsilon_values = [1e-6],
        delta_values   = [1e-3],
        I_values       = [200],
        Tmin_values    = [0.1],
        tol=1e-2,
        Nexec=Nexec,
        N_rep=N_rep_timing,
        csv_path=csv_path_da,
        best_record_path=best_record_path_da,
    )

    print("\n===== Top 10 combinações (por SR) =====")
    cols_show = ["T0","alpha","epsilon","delta","I","Tmin",
                 "SR","MBF","AES","D_best","tempo_medio_s"]
    cols_ok = [c for c in cols_show if c in df_results.columns]
    print(df_results[cols_ok].head(10).to_string())

    if best_record is not None:
        print("\n===== Melhor combinação =====")
        for k in ["T0","alpha","epsilon","delta","I","Tmin",
                  "SR","MBF","AES","D_best"]:
            if k in best_record:
                print(f"  {k} = {best_record[k]}")
        plot_best_solution_da(data_vectors, cluster_centers, best_record)
        medir_tempo_da(data_vectors, NC_number, best_record, N_rep=N_rep_timing)