# Estratégias de Evolução (ES)
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

# =========================================================
# 0. Diretório de saída local
# =========================================================
# local_path é definido dinamicamente no __main__ com base em NC_number
local_path = None  # será atualizado em __main__
# csv_path_es e best_record_path_es são definidos em __main__
csv_path_es = None
best_record_path_es = None
print("Resultados ES serão salvos em:")
print(csv_path_es)
print(best_record_path_es)

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
# 2. Função custo "hard" — mesma do DA
# =========================================================
def J_hard(X, Y):
    diff  = X[:, :, None] - Y[:, None, :]
    dist2 = np.sum(diff**2, axis=0)
    return np.mean(np.min(dist2, axis=0))

# =========================================================
# 3. Uma execução do ES
#
# Parâmetros:
#   Nind        — tamanho da população (mu)
#   Npais       — pais sorteados por geração (seleção uniforme)
#   Nfilhos     — filhos gerados por geração (lambda)
#                  dica: usar Nfilhos = 7*Nind (recomendação literatura)
#   Nsob        — sobreviventes — estratégia (mu, lambda)
#   Nger        — número máximo de gerações
#   epson0      — sigma_min
#   tau1        — fator global de aprendizado do sigma (None = 1/sqrt(2*Nd))
#   tau2        — fator individual de aprendizado do sigma (None = 1/sqrt(2*sqrt(Nd)))
#   sigma0_low  — limite inferior da inicialização dos sigmas
#   sigma0_high — limite superior da inicialização dos sigmas
#   clip_val    — limita centróides ao intervalo [-clip_val, +clip_val] após mutação
#   patience    — para se sem melhora por N gerações consecutivas
#   prec        — probabilidade de recombinação (0 a 1)
#   pmut        — probabilidade de mutação (0 a 1)
#   tol         — tolerância para considerar sucesso (|J - J_ref| <= tol)
#   J_ref       — custo de referência (centros verdadeiros)
# =========================================================
def run_es(
    data_vectors,
    NC,
    Nind=50,
    Npais=40,
    Nfilhos=300,
    Nsob=50,
    Nger=800,
    epson0=1e-8,
    tau1=None,
    tau2=None,
    sigma0_low=0.1,
    sigma0_high=0.5,
    clip_val=None,
    patience=50,
    prec=1.0,
    pmut=1.0,
    tol=1e-2,
    J_ref=None,
    init_seed=0
):
    start_time = time.perf_counter()
    rng  = np.random.default_rng(init_seed)
    Nd   = 3 * NC

    if tau1 is None:
        tau1 = 1.0 / np.sqrt(2 * Nd)
    if tau2 is None:
        tau2 = 1.0 / np.sqrt(2 * np.sqrt(Nd))

    # Inicialização — igual ao DA: N(0,1)
    x_ini     = rng.standard_normal((Nind, Nd))
    sigma_ini = sigma0_low + (sigma0_high - sigma0_low) * rng.random((Nind, Nd))
    ind = np.hstack([x_ini, sigma_ini])

    def custo_batch(pop):
        n_pop = len(pop)
        C     = pop[:, :Nd].reshape(n_pop, 3, NC).transpose(0, 2, 1)
        dv    = data_vectors.T
        diff  = C[:, :, None, :] - dv[None, None, :, :]
        dist2 = np.sum(diff**2, axis=3)
        return np.mean(np.min(dist2, axis=1), axis=1)

    J_ini  = custo_batch(ind)
    imin   = np.argmin(J_ini)
    D_best = J_ini[imin]
    Y_best = ind[imin, :Nd].reshape(3, NC).copy()

    history_D     = np.zeros(Nger)
    history_Dbest = np.zeros(Nger)
    history_time  = np.zeros(Nger)   # tempo acumulado por geração
    sem_melhora   = 0
    ncalls        = Nind              # avaliações iniciais
    isuc          = 0                 # flag de sucesso
    ncalls_suc    = 0                 # avaliações até o sucesso

    for g in range(Nger):

        t_ger = time.perf_counter()

        # Seleção de pais: sorteio aleatório uniforme
        # usa len(ind) e não Nind: a partir da 2ª geração ind tem Nsob linhas
        idx_pais = rng.integers(len(ind), size=Npais)
        pais     = ind[idx_pais]

        # Recombinação (com probabilidade prec)
        i1s  = rng.integers(Npais, size=Nfilhos)
        i2s  = rng.integers(Npais, size=Nfilhos)
        mask_rec = rng.random(Nfilhos) < prec   # quais filhos recombinar
        mask_var = rng.random((Nfilhos, Nd)) < 0.5

        filhos = np.zeros((Nfilhos, 2 * Nd))
        # filhos que recombinaram: discreta nas variáveis
        filhos[mask_rec, :Nd]  = np.where(
            mask_var[mask_rec],
            pais[i1s[mask_rec], :Nd],
            pais[i2s[mask_rec], :Nd]
        )
        # filhos que recombinaram: intermediária nos sigmas
        filhos[mask_rec, Nd:]  = 0.5 * (
            pais[i1s[mask_rec], Nd:] + pais[i2s[mask_rec], Nd:]
        )
        # filhos que NÃO recombinaram: cópia do pai 1
        filhos[~mask_rec, :Nd] = pais[i1s[~mask_rec], :Nd]
        filhos[~mask_rec, Nd:] = pais[i1s[~mask_rec], Nd:]

        # Mutação (com probabilidade pmut por filho)
        mask_mut = rng.random(Nfilhos) < pmut
        if mask_mut.any():
            r_comum = rng.standard_normal((Nfilhos, 1))
            r_comp  = rng.standard_normal((Nfilhos, Nd))
            r_mut   = rng.standard_normal((Nfilhos, Nd))

            shat = filhos[:, Nd:] * np.exp(tau1 * r_comum + tau2 * r_comp)
            shat = np.maximum(shat, epson0)

            filhos[mask_mut, :Nd] += shat[mask_mut] * r_mut[mask_mut]
            filhos[mask_mut, Nd:]  = shat[mask_mut]

        # Clip
        if clip_val is not None:
            filhos[:, :Nd] = np.clip(filhos[:, :Nd], -clip_val, clip_val)

        # Avaliação dos filhos
        J_fil  = custo_batch(filhos)
        ncalls += Nfilhos

        # Seleção (mu, lambda)
        idx_sort = np.argsort(J_fil)
        ind      = filhos[idx_sort[:Nsob]]

        if J_fil[idx_sort[0]] < D_best:
            D_best      = J_fil[idx_sort[0]]
            Y_best      = ind[0, :Nd].reshape(3, NC).copy()
            sem_melhora = 0
        else:
            sem_melhora += 1

        history_D[g]     = float(np.mean(J_fil))
        history_Dbest[g] = D_best
        history_time[g]  = time.perf_counter() - start_time

        # Verifica sucesso
        if isuc == 0 and J_ref is not None:
            if abs(D_best - J_ref) <= tol:
                isuc       = 1
                ncalls_suc = ncalls

        # Critério de parada antecipada
        if sem_melhora >= patience:
            history_D     = history_D[:g+1]
            history_Dbest = history_Dbest[:g+1]
            history_time  = history_time[:g+1]
            break

    elapsed = time.perf_counter() - start_time
    print(f"    Tempo de execução: {elapsed:.2f} s | D_best={D_best:.6f} | isuc={isuc}")

    return D_best, Y_best, history_D, history_Dbest, history_time, isuc, ncalls_suc

# =========================================================
# 4. Loop de 100 execuções — calcula SR, MBF, AES
#    (igual à metodologia da Ana Cláudia)
# =========================================================
def run_100_execucoes(
    data_vectors,
    NC,
    J_ref,
    Nind, Npais, Nfilhos, Nsob, Nger,
    epson0, tau1, tau2,
    sigma0_low, sigma0_high,
    clip_val, patience,
    prec, pmut,
    tol=1e-2,
    Nexec=100,
    base_seed=0
):
    """
    Executa o ES Nexec vezes com seeds independentes e calcula:
      SR  — taxa de sucesso (% execuções com |J - J_ref| <= tol)
      MBF — média do melhor custo encontrado em cada execução
      AES — número médio de avaliações até o sucesso (só nas bem-sucedidas)
    """
    J_runs    = []
    isuc_runs = []
    aes_runs  = []
    best_hist_D     = None
    best_hist_Dbest = None
    best_hist_time  = None
    best_J    = np.inf

    for i in range(Nexec):
        seed = base_seed + i
        D_best, Y_best, hist_D, hist_Dbest, hist_time, isuc, ncalls_suc = run_es(
            data_vectors=data_vectors, NC=NC,
            Nind=Nind, Npais=Npais, Nfilhos=Nfilhos, Nsob=Nsob,
            Nger=Nger, epson0=epson0, tau1=tau1, tau2=tau2,
            sigma0_low=sigma0_low, sigma0_high=sigma0_high,
            clip_val=clip_val, patience=patience,
            prec=prec, pmut=pmut,
            tol=tol, J_ref=J_ref,
            init_seed=seed
        )
        J_runs.append(D_best)
        isuc_runs.append(isuc)
        if isuc:
            aes_runs.append(ncalls_suc)
        if D_best < best_J:
            best_J          = D_best
            best_hist_D     = hist_D.copy()
            best_hist_Dbest = hist_Dbest.copy()
            best_hist_time  = hist_time.copy()

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{Nexec} execuções concluídas")

    SR  = float(np.mean(isuc_runs))
    MBF = float(np.mean(J_runs))
    AES = float(np.mean(aes_runs)) if aes_runs else np.nan

    print(f"  SR={SR*100:.1f}%  MBF={MBF:.6f}  AES={AES:.0f}")

    return {
        "SR":  SR,
        "MBF": MBF,
        "AES": AES,
        "J_best":         best_J,
        "best_hist_D":    best_hist_D,
        "best_hist_Dbest":best_hist_Dbest,
        "best_hist_time": best_hist_time,
    }

# =========================================================
# 5. Grid search para ES
# =========================================================
def grid_search_es(
    data_vectors,
    cluster_centers,
    NC,
    Nind_values,
    Npais_values,
    Nfilhos_values,
    Nsob_values,
    Nger_values,
    epson0_values,
    tau1_values        = [None],
    tau2_values        = [None],
    sigma0_low_values  = [0.1],
    sigma0_high_values = [0.5],
    clip_val_values    = [None],
    patience_values    = [50],
    prec_values        = [1.0],
    pmut_values        = [1.0],
    tol=1e-2,
    Nexec=100,
    N_rep=10,
    csv_path=csv_path_es,
    best_record_path=best_record_path_es
):
    J_global_ref = J_hard(cluster_centers, data_vectors)
    Nd = 3 * NC

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        key_cols  = ["Nind", "Npais", "Nfilhos", "Nsob", "Nger",
                     "epson0", "tau1", "tau2",
                     "sigma0_low", "sigma0_high", "clip_val",
                     "patience", "prec", "pmut"]
        available = [c for c in key_cols if c in df_existing.columns]
        tested    = set(tuple(row) for row in df_existing[available].values)
        print(f"{len(df_existing)} linhas já existentes no CSV — serão puladas.")
    else:
        df_existing = pd.DataFrame()
        tested      = set()

    combinations = list(itertools.product(
        Nind_values, Npais_values, Nfilhos_values, Nsob_values,
        Nger_values, epson0_values,
        tau1_values, tau2_values,
        sigma0_low_values, sigma0_high_values,
        clip_val_values, patience_values,
        prec_values, pmut_values
    ))
    total = len(combinations)
    print(f"Total de combinações possíveis: {total}")

    best_global_value = np.inf
    if os.path.exists(best_record_path):
        old_best = np.load(best_record_path, allow_pickle=True)
        best_global_value = float(old_best["J_best"])
        print(f"Melhor global prévio encontrado: {best_global_value:.6f}")

    best_record  = None
    t_grid_start = time.perf_counter()

    for idx, (Nind, Npais, Nfilhos, Nsob, Nger, epson0,
              tau1, tau2, sigma0_low, sigma0_high,
              clip_val, patience,
              prec, pmut) in enumerate(combinations, start=1):

        tau1_real = tau1 if tau1 is not None else 1.0 / np.sqrt(2 * Nd)
        tau2_real = tau2 if tau2 is not None else 1.0 / np.sqrt(2 * np.sqrt(Nd))

        # Validações
        if Nsob > Nind:
            print(f"[{idx}/{total}] Pulando: Nsob={Nsob} > Nind={Nind}")
            continue
        pop_minima = min(Nind, Nsob)
        if Npais > pop_minima:
            print(f"[{idx}/{total}] Pulando: Npais={Npais} > min(Nind,Nsob)={pop_minima}")
            continue

        comb_key = (Nind, Npais, Nfilhos, Nsob, Nger, epson0,
                    tau1, tau2, sigma0_low, sigma0_high,
                    clip_val, patience, prec, pmut)
        if comb_key in tested:
            print(f"[{idx}/{total}] Pulando combinação já calculada.")
            continue

        # ETA
        elapsed_grid = time.perf_counter() - t_grid_start
        eta_str = (str(datetime.timedelta(
                        seconds=int(elapsed_grid / (idx-1) * (total-idx+1))))
                   if idx > 1 else "calculando...")

        print(f"\n{'='*50}")
        print(f"Combinação {idx}/{total}  |  ETA: {eta_str}")
        print(f"Nind={Nind}, Npais={Npais}, Nfilhos={Nfilhos}, Nsob={Nsob}, "
              f"Nger={Nger}, epson0={epson0}")
        print(f"tau1={tau1_real:.5f}, tau2={tau2_real:.5f}, "
              f"sigma0=[{sigma0_low},{sigma0_high}], clip={clip_val}, "
              f"patience={patience}, prec={prec}, pmut={pmut}")

        # 100 execuções para SR / MBF / AES
        metrics = run_100_execucoes(
            data_vectors=data_vectors, NC=NC, J_ref=J_global_ref,
            Nind=Nind, Npais=Npais, Nfilhos=Nfilhos, Nsob=Nsob,
            Nger=Nger, epson0=epson0, tau1=tau1, tau2=tau2,
            sigma0_low=sigma0_low, sigma0_high=sigma0_high,
            clip_val=clip_val, patience=patience,
            prec=prec, pmut=pmut,
            tol=tol, Nexec=Nexec, base_seed=0
        )

        # Medição de tempo (N_rep repetições, seed=0)
        print(f"  -> Medindo tempo ({N_rep} repetições)...")
        t_runs = []
        for _ in range(N_rep):
            t0 = time.perf_counter()
            run_es(
                data_vectors=data_vectors, NC=NC,
                Nind=Nind, Npais=Npais, Nfilhos=Nfilhos, Nsob=Nsob,
                Nger=Nger, epson0=epson0, tau1=tau1, tau2=tau2,
                sigma0_low=sigma0_low, sigma0_high=sigma0_high,
                clip_val=clip_val, patience=patience,
                prec=prec, pmut=pmut,
                tol=tol, J_ref=J_global_ref, init_seed=0
            )
            t_runs.append(time.perf_counter() - t0)
        tempo_medio = float(np.mean(t_runs))
        tempo_std   = float(np.std(t_runs))
        print(f"  -> Tempo médio: {tempo_medio:.3f} s | Std: {tempo_std:.3f} s")

        # Salva arquivos
        combo_tag = (
            f"Nind{Nind}_Np{Npais}_Nfil{Nfilhos}_Nsob{Nsob}"
            f"_Nger{Nger}_eps{epson0}"
            f"_t1{tau1}_t2{tau2}"
            f"_s0{sigma0_low}_{sigma0_high}_clip{clip_val}"
            f"_pat{patience}_pr{prec}_pm{pmut}"
        ).replace(".", "p").replace("None", "def")

        centroids_path     = os.path.join(local_path, f"es_centroids_{combo_tag}.npy")
        history_D_path     = os.path.join(local_path, f"es_history_D_{combo_tag}.npy")
        history_Dbest_path = os.path.join(local_path, f"es_history_Dbest_{combo_tag}.npy")
        history_time_path  = os.path.join(local_path, f"es_history_time_{combo_tag}.npy")

        np.save(history_D_path,     metrics["best_hist_D"])
        np.save(history_Dbest_path, metrics["best_hist_Dbest"])
        np.save(history_time_path,  metrics["best_hist_time"])

        row = {
            "timestamp":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Nind":               Nind,
            "Npais":              Npais,
            "Nfilhos":            Nfilhos,
            "Nsob":               Nsob,
            "Nger":               Nger,
            "epson0":             epson0,
            "tau1":               tau1,
            "tau2":               tau2,
            "tau1_real":          tau1_real,
            "tau2_real":          tau2_real,
            "sigma0_low":         sigma0_low,
            "sigma0_high":        sigma0_high,
            "clip_val":           clip_val,
            "patience":           patience,
            "prec":               prec,
            "pmut":               pmut,
            "tol":                tol,
            "Nexec":              Nexec,
            "SR":                 metrics["SR"],
            "MBF":                metrics["MBF"],
            "AES":                metrics["AES"],
            "J_best":             metrics["J_best"],
            "J_global_reference": float(J_global_ref),
            "tempo_medio_s":      tempo_medio,
            "tempo_std_s":        tempo_std,
            "history_D_file":     history_D_path,
            "history_Dbest_file": history_Dbest_path,
            "history_time_file":  history_time_path,
        }

        df_row = pd.DataFrame([row])
        write_header = not os.path.exists(csv_path)
        df_row.to_csv(csv_path, mode="a", header=write_header, index=False)
        print(f"  -> SR={metrics['SR']*100:.1f}% | MBF={metrics['MBF']:.6f} | "
              f"AES={metrics['AES']:.0f} | salvo no CSV")

        if metrics["J_best"] < best_global_value:
            best_global_value = metrics["J_best"]
            best_record = {
                "Nind": Nind, "Npais": Npais, "Nfilhos": Nfilhos,
                "Nsob": Nsob, "Nger": Nger, "epson0": epson0,
                "tau1": tau1, "tau2": tau2,
                "tau1_real": tau1_real, "tau2_real": tau2_real,
                "sigma0_low": sigma0_low, "sigma0_high": sigma0_high,
                "clip_val": clip_val, "patience": patience,
                "prec": prec, "pmut": pmut,
                "J_best":      metrics["J_best"],
                "SR":          metrics["SR"],
                "MBF":         metrics["MBF"],
                "AES":         metrics["AES"],
                "best_hist_D":     metrics["best_hist_D"],
                "best_hist_Dbest": metrics["best_hist_Dbest"],
                "best_hist_time":  metrics["best_hist_time"],
            }
            np.savez(
                best_record_path,
                Nind=Nind, Npais=Npais, Nfilhos=Nfilhos,
                Nsob=Nsob, Nger=Nger, epson0=epson0,
                tau1=str(tau1), tau2=str(tau2),
                tau1_real=tau1_real, tau2_real=tau2_real,
                sigma0_low=sigma0_low, sigma0_high=sigma0_high,
                clip_val=str(clip_val), patience=patience,
                prec=prec, pmut=pmut,
                J_best=metrics["J_best"],
                SR=metrics["SR"], MBF=metrics["MBF"], AES=metrics["AES"],
                best_hist_D=metrics["best_hist_D"],
                best_hist_Dbest=metrics["best_hist_Dbest"],
                best_hist_time=metrics["best_hist_time"],
            )
            print(f"  -> NOVO melhor global: {best_global_value:.6f} — salvo")

    print(f"\nGrid search concluído. Tempo total: "
          f"{str(datetime.timedelta(seconds=int(time.perf_counter()-t_grid_start)))}")

    df_results = (
        pd.read_csv(csv_path)
        .sort_values(by="SR", ascending=False)   # ordena por SR primeiro
        .reset_index(drop=True)
    )

    if best_record is None and os.path.exists(best_record_path):
        loaded = np.load(best_record_path, allow_pickle=True)
        best_record = {k: loaded[k] for k in loaded.files}

    return df_results, best_record

# =========================================================
# 6. Plot ES — dois gráficos de convergência
#    (eixo x por gerações E por tempo)
# =========================================================
def plot_best_solution_es(data_vectors, cluster_centers, best_record):

    Xbest          = best_record["best_X"] if "best_X" in best_record else None
    history_D      = np.asarray(best_record["best_hist_D"])
    history_Dbest  = np.asarray(best_record["best_hist_Dbest"])
    history_time   = np.asarray(best_record["best_hist_time"])
    J_ref          = J_hard(cluster_centers, data_vectors)

    # ── Gráfico 1: eixo x = gerações ──────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(history_D,     'r-', label='Custo médio da geração')
    plt.plot(history_Dbest, 'k-', label='Melhor custo acumulado (D_best)')
    plt.axhline(y=J_ref, color='b', linestyle='--', linewidth=1.0,
                label='Custo com centros verdadeiros')
    plt.grid(); plt.xlabel('Geração'); plt.ylabel('Custo')
    plt.title('Estratégia de Evolução (ES) — convergência por geração')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(local_path, "es_convergencia_geracao.png"), dpi=150)
    plt.show()

    # ── Gráfico 2: eixo x = tempo (s) — mesmo estilo da Ana Cláudia ──
    plt.figure(figsize=(10, 6))
    plt.plot(history_time, history_D,     'r-', label='Custo médio da geração')
    plt.plot(history_time, history_Dbest, 'k-', label='Melhor custo acumulado (D_best)')
    plt.axhline(y=J_ref, color='b', linestyle='--', linewidth=1.0,
                label='Custo com centros verdadeiros')
    plt.grid(); plt.xlabel('Tempo (s)'); plt.ylabel('Custo')
    plt.title('Estratégia de Evolução (ES) — convergência por tempo')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(local_path, "es_convergencia_tempo.png"), dpi=150)
    plt.show()

    # ── Gráfico 3D ────────────────────────────────────────────────────
    if Xbest is not None:
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(data_vectors[0,:], data_vectors[1,:], data_vectors[2,:],
                   c='k', s=8, alpha=0.25, depthshade=False, label='Dados')
        ax.scatter(Xbest[0,:], Xbest[1,:], Xbest[2,:],
                   c='red', s=220, marker='o',
                   edgecolors='white', linewidths=1.8, depthshade=False,
                   label='Centróides ES')
        ax.scatter(cluster_centers[0,:], cluster_centers[1,:], cluster_centers[2,:],
                   c='blue', s=260, marker='x', linewidths=2.5, depthshade=False,
                   label='Centros verdadeiros')
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
        ax.set_title("ES - Clustering 3D")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(local_path, "es_clustering_3d.png"), dpi=150)
        plt.show()

# =========================================================
# 7. Medição de tempo de execução (chamada explícita)
# =========================================================
def medir_tempo_es(data_vectors, NC, best_record, N_rep=10):
    tempos = []

    def parse_none(v):
        if v is None or (isinstance(v, str) and v in ("None", "def")):
            return None
        return float(v)

    Nind        = int(best_record["Nind"])
    Npais       = int(best_record["Npais"])
    Nfilhos     = int(best_record["Nfilhos"])
    Nsob        = int(best_record["Nsob"])
    Nger        = int(best_record["Nger"])
    epson0      = float(best_record["epson0"])
    tau1        = parse_none(best_record["tau1"])
    tau2        = parse_none(best_record["tau2"])
    sigma0_low  = float(best_record["sigma0_low"])
    sigma0_high = float(best_record["sigma0_high"])
    clip_val    = parse_none(best_record["clip_val"])
    patience    = int(best_record["patience"])
    prec        = float(best_record["prec"])
    pmut        = float(best_record["pmut"])

    J_ref = J_hard(
        np.load(best_record_path_es.replace("best_record_es.npz", ""))
        if False else np.zeros((3, NC)),
        data_vectors
    ) if False else None

    print(f"\nMedindo tempo de execução ({N_rep} repetições)...")
    for rep in range(N_rep):
        t0 = time.perf_counter()
        run_es(
            data_vectors=data_vectors, NC=NC,
            Nind=Nind, Npais=Npais, Nfilhos=Nfilhos, Nsob=Nsob,
            Nger=Nger, epson0=epson0, tau1=tau1, tau2=tau2,
            sigma0_low=sigma0_low, sigma0_high=sigma0_high,
            clip_val=clip_val, patience=patience,
            prec=prec, pmut=pmut, init_seed=0
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

    NC_number = 4   # ← mude aqui para 4, 8, 16, etc.

    # Diretório de saída automático baseado no NC_number
    local_path = os.path.expanduser(f"~/cpe723_es_nc{NC_number}")
    os.makedirs(local_path, exist_ok=True)
    csv_path_es = os.path.join(local_path, "grid_search_es_results.csv")
    best_record_path_es = os.path.join(local_path, "best_record_es.npz")
    print(f"Resultados salvos em: {local_path}")
    data_vectors, cluster_centers = generate_data_r3(P=100, NC=NC_number, sigma=0.1, seed=1)
    J_ref = J_hard(cluster_centers, data_vectors)
    print(f"Custo com centros verdadeiros: {J_ref:.6f}")

    N_rep_timing = 10   # repetições para medir tempo
    Nexec        = 100  # execuções por combinação (para SR/MBF/AES)

    # ---------------------------------------------------------
    # Fase 1: sem clip — explora parâmetros populacionais,
    #         prec e pmut
    # ---------------------------------------------------------
    df_results, best_record = grid_search_es(
        data_vectors=data_vectors,
        cluster_centers=cluster_centers,
        NC=NC_number,
        Nind_values        = [50],
        Npais_values       = [40],
        Nfilhos_values     = [300],
        Nsob_values        = [50],
        Nger_values        = [800],
        epson0_values      = [1e-8],
        tau1_values        = [None],
        tau2_values        = [None],
        sigma0_low_values  = [0.1],
        sigma0_high_values = [0.5],
        clip_val_values    = [None],
        patience_values    = [50],
        prec_values        = [0.7, 0.9, 1.0],   # varia prec
        pmut_values        = [0.7, 0.9, 1.0],   # varia pmut
        tol=1e-2,
        Nexec=Nexec,
        N_rep=N_rep_timing,
        csv_path=csv_path_es,
        best_record_path=best_record_path_es,
    )

    # ---------------------------------------------------------
    # Fase 2: com clip — descomente após concluir Fase 1
    # ---------------------------------------------------------
    # df_results, best_record = grid_search_es(
    #     data_vectors=data_vectors,
    #     cluster_centers=cluster_centers,
    #     NC=NC_number,
    #     Nind_values        = [50],
    #     Npais_values       = [40],
    #     Nfilhos_values     = [300],
    #     Nsob_values        = [50],
    #     Nger_values        = [800],
    #     epson0_values      = [1e-8],
    #     tau1_values        = [None],
    #     tau2_values        = [None],
    #     sigma0_low_values  = [0.1],
    #     sigma0_high_values = [0.5],
    #     clip_val_values    = [3.0, 5.0],
    #     patience_values    = [50],
    #     prec_values        = [1.0],
    #     pmut_values        = [1.0],
    #     tol=1e-2,
    #     Nexec=Nexec,
    #     N_rep=N_rep_timing,
    # )

    print("\n===== Top 10 combinações (por SR) =====")
    cols_show = ["Nind","Npais","Nfilhos","Nsob","Nger",
                 "prec","pmut","patience","clip_val",
                 "SR","MBF","AES","J_best","tempo_medio_s"]
    cols_ok = [c for c in cols_show if c in df_results.columns]
    print(df_results[cols_ok].head(10).to_string())

    if best_record is not None:
        print("\n===== Melhor combinação =====")
        for k in ["Nind","Npais","Nfilhos","Nsob","Nger",
                  "prec","pmut","patience","SR","MBF","AES","J_best"]:
            if k in best_record:
                print(f"  {k} = {best_record[k]}")

        plot_best_solution_es(data_vectors, cluster_centers, best_record)