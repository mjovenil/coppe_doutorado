# Algoritmo Híbrido ES+DA
# Versão local — Linux / VS Code
#
# Ideia: a cada geração do ES, após a recombinação e antes da
# mutação gaussiana, aplica-se uma iteração do DA sobre os
# centróides de cada filho. O DA fornece uma direção de melhora
# determinística (gradiente implícito), que orienta a busca do ES
# em vez de deixá-la completamente aleatória.
#
# Fluxo por geração:
#   1. Seleção de pais         (igual ao ES)
#   2. Recombinação            (igual ao ES)
#   3. Passo DA                (novo — orienta os centróides)
#   4. Mutação gaussiana       (igual ao ES — mantém diversidade)
#   5. Seleção (mu, lambda)    (igual ao ES)
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
# csv_path_hib e best_record_path_hib são definidos em __main__
csv_path_hib = None
best_record_path_hib = None
print("Resultados Híbrido serão salvos em:")
# csv_path_hib e best_record_path_hib serão impressos em __main__

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
# 2. Função custo "hard"
# =========================================================
def J_hard(X, Y):
    diff  = X[:, :, None] - Y[:, None, :]
    dist2 = np.sum(diff**2, axis=0)
    return np.mean(np.min(dist2, axis=0))

# =========================================================
# 3. Passo DA — uma iteração do DA sobre um conjunto de
#    centróides Y, retorna os centróides atualizados Y_new.
#
#    Este é o "gradiente implícito" que orienta o ES:
#    Y_new = argmin_Y D(Y) dado Y atual e temperatura T.
#
#    Para T alto: atualização suave, centróides se movem pouco.
#    Para T baixo: atualização mais agressiva, convergência local.
# =========================================================
def passo_da(X, Y, T):
    """
    X: dados (M, N)
    Y: centróides atuais (M, K)
    T: temperatura do DA
    Retorna Y_new: centróides após uma iteração do DA
    """
    diff = X[:, :, None] - Y[:, None, :]   # (M, N, K)
    d    = np.sum(diff**2, axis=0).T        # (K, N)
    p    = np.exp(-d / T)
    Zx   = np.sum(p, axis=0)               # (N,)
    p   /= Zx                              # normaliza
    weights = np.sum(p, axis=1)            # (K,)
    Y_new   = (X @ p.T) / weights         # (M, K)
    return Y_new

# =========================================================
# 4. Uma execução do Híbrido ES+DA
#
# Parâmetros novos em relação ao ES puro:
#   T_da     — temperatura usada no passo DA por geração.
#               Valores altos: guiamento suave (exploração).
#               Valores baixos: guiamento forte (explotação).
#               default: 1.0
#   alpha_da — fator de mistura entre passo DA e posição atual:
#               Y_guiado = alpha_da * Y_da + (1-alpha_da) * Y_atual
#               alpha_da=1.0: usa completamente o passo DA.
#               alpha_da=0.0: ignora o DA (equivale ao ES puro).
#               default: 1.0
# =========================================================
def run_hibrido(
    data_vectors,
    NC,
    Nind=100,
    Npais=100,
    Nfilhos=700,
    Nsob=100,
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
    T_da=1.0,
    alpha_da=1.0,
    tol=1e-2,
    J_ref=None,
    init_seed=0
):
    start_time = time.perf_counter()
    rng  = np.random.default_rng(init_seed)
    Nd   = 3 * NC
    X    = data_vectors                    # (3, N)

    if tau1 is None:
        tau1 = 1.0 / np.sqrt(2 * Nd)
    if tau2 is None:
        tau2 = 1.0 / np.sqrt(2 * np.sqrt(Nd))

    # Inicialização — igual ao ES/DA: N(0,1)
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
    history_time  = np.zeros(Nger)
    sem_melhora   = 0
    ncalls        = Nind
    isuc          = 0
    ncalls_suc    = 0

    for g in range(Nger):

        # ── Seleção de pais ───────────────────────────────────────────
        idx_pais = rng.integers(len(ind), size=Npais)
        pais     = ind[idx_pais]

        # ── Recombinação ──────────────────────────────────────────────
        i1s      = rng.integers(Npais, size=Nfilhos)
        i2s      = rng.integers(Npais, size=Nfilhos)
        mask_rec = rng.random(Nfilhos) < prec
        mask_var = rng.random((Nfilhos, Nd)) < 0.5

        filhos = np.zeros((Nfilhos, 2 * Nd))
        filhos[mask_rec,  :Nd] = np.where(
            mask_var[mask_rec],
            pais[i1s[mask_rec], :Nd],
            pais[i2s[mask_rec], :Nd]
        )
        filhos[mask_rec,  Nd:] = 0.5 * (
            pais[i1s[mask_rec], Nd:] + pais[i2s[mask_rec], Nd:]
        )
        filhos[~mask_rec, :Nd] = pais[i1s[~mask_rec], :Nd]
        filhos[~mask_rec, Nd:] = pais[i1s[~mask_rec], Nd:]

        # ── Passo DA — orienta os centróides de cada filho ────────────
        # Aplica uma iteração do DA sobre os centróides de cada filho,
        # obtendo uma direção de melhora determinística.
        # Y_guiado = alpha_da * Y_da + (1-alpha_da) * Y_atual
        if alpha_da > 0.0:
            for j in range(Nfilhos):
                Y_atual = filhos[j, :Nd].reshape(3, NC)
                Y_da    = passo_da(X, Y_atual, T_da)
                Y_guiado = alpha_da * Y_da + (1.0 - alpha_da) * Y_atual
                filhos[j, :Nd] = Y_guiado.flatten()

        # ── Mutação gaussiana com auto-adaptação dos sigmas ───────────
        mask_mut = rng.random(Nfilhos) < pmut
        if mask_mut.any():
            r_comum = rng.standard_normal((Nfilhos, 1))
            r_comp  = rng.standard_normal((Nfilhos, Nd))
            r_mut   = rng.standard_normal((Nfilhos, Nd))

            shat = filhos[:, Nd:] * np.exp(tau1 * r_comum + tau2 * r_comp)
            shat = np.maximum(shat, epson0)

            filhos[mask_mut, :Nd] += shat[mask_mut] * r_mut[mask_mut]
            filhos[mask_mut, Nd:]  = shat[mask_mut]

        # ── Clip ──────────────────────────────────────────────────────
        if clip_val is not None:
            filhos[:, :Nd] = np.clip(filhos[:, :Nd], -clip_val, clip_val)

        # ── Avaliação e seleção (mu, lambda) ──────────────────────────
        J_fil  = custo_batch(filhos)
        ncalls += Nfilhos

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

        if isuc == 0 and J_ref is not None:
            if abs(D_best - J_ref) <= tol:
                isuc       = 1
                ncalls_suc = ncalls

        if sem_melhora >= patience:
            history_D     = history_D[:g+1]
            history_Dbest = history_Dbest[:g+1]
            history_time  = history_time[:g+1]
            break

    elapsed = time.perf_counter() - start_time
    print(f"    Tempo: {elapsed:.2f} s | J_best={D_best:.6f} | isuc={isuc}")

    return D_best, Y_best, history_D, history_Dbest, history_time, isuc, ncalls_suc

# =========================================================
# 5. Loop de 100 execuções — SR, MBF, AES
# =========================================================
def run_100_execucoes_hibrido(
    data_vectors, NC, J_ref,
    Nind, Npais, Nfilhos, Nsob, Nger,
    epson0, tau1, tau2,
    sigma0_low, sigma0_high,
    clip_val, patience, prec, pmut,
    T_da, alpha_da,
    tol=1e-2, Nexec=100, base_seed=0
):
    D_runs = []; isuc_runs = []; aes_runs = []
    best_hist_D = best_hist_Dbest = best_hist_time = None
    best_D = np.inf

    for i in range(Nexec):
        D_best, Y_best, hist_D, hist_Dbest, hist_time, isuc, ncalls_suc = run_hibrido(
            data_vectors=data_vectors, NC=NC,
            Nind=Nind, Npais=Npais, Nfilhos=Nfilhos, Nsob=Nsob,
            Nger=Nger, epson0=epson0, tau1=tau1, tau2=tau2,
            sigma0_low=sigma0_low, sigma0_high=sigma0_high,
            clip_val=clip_val, patience=patience,
            prec=prec, pmut=pmut,
            T_da=T_da, alpha_da=alpha_da,
            tol=tol, J_ref=J_ref, init_seed=base_seed + i
        )
        D_runs.append(D_best); isuc_runs.append(isuc)
        if isuc:
            aes_runs.append(ncalls_suc)
        if D_best < best_D:
            best_D         = D_best
            best_hist_D    = hist_D.copy()
            best_hist_Dbest= hist_Dbest.copy()
            best_hist_time = hist_time.copy()

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{Nexec} execuções concluídas")

    SR  = float(np.mean(isuc_runs))
    MBF = float(np.mean(D_runs))
    AES = float(np.mean(aes_runs)) if aes_runs else np.nan
    print(f"  SR={SR*100:.1f}%  MBF={MBF:.6f}  AES={AES:.0f}")

    return {
        "SR": SR, "MBF": MBF, "AES": AES, "J_best": best_D,
        "best_hist_D": best_hist_D,
        "best_hist_Dbest": best_hist_Dbest,
        "best_hist_time": best_hist_time,
    }

# =========================================================
# 6. Grid search para o Híbrido
# =========================================================
def grid_search_hibrido(
    data_vectors, cluster_centers, NC,
    Nind_values, Npais_values, Nfilhos_values, Nsob_values,
    Nger_values, epson0_values,
    tau1_values        = [None],
    tau2_values        = [None],
    sigma0_low_values  = [0.1],
    sigma0_high_values = [0.5],
    clip_val_values    = [None],
    patience_values    = [50],
    prec_values        = [1.0],
    pmut_values        = [1.0],
    T_da_values        = [1.0],
    alpha_da_values    = [1.0],
    tol=1e-2, Nexec=100, N_rep=10,
    csv_path=csv_path_hib,
    best_record_path=best_record_path_hib
):
    J_global_ref = J_hard(cluster_centers, data_vectors)
    Nd = 3 * NC

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        key_cols  = ["Nind","Npais","Nfilhos","Nsob","Nger","epson0",
                     "tau1","tau2","sigma0_low","sigma0_high","clip_val",
                     "patience","prec","pmut","T_da","alpha_da"]
        available = [c for c in key_cols if c in df_existing.columns]
        tested    = set(tuple(row) for row in df_existing[available].values)
        print(f"{len(df_existing)} linhas já existentes no CSV — serão puladas.")
    else:
        tested = set()

    combinations = list(itertools.product(
        Nind_values, Npais_values, Nfilhos_values, Nsob_values,
        Nger_values, epson0_values,
        tau1_values, tau2_values,
        sigma0_low_values, sigma0_high_values,
        clip_val_values, patience_values,
        prec_values, pmut_values,
        T_da_values, alpha_da_values
    ))
    total = len(combinations)
    print(f"Total de combinações possíveis: {total}")

    best_global_value = np.inf
    if os.path.exists(best_record_path):
        old_best = np.load(best_record_path, allow_pickle=True)
        best_global_value = float(old_best["J_best"])
        print(f"Melhor global prévio: {best_global_value:.6f}")

    best_record  = None
    t_grid_start = time.perf_counter()

    for idx, (Nind, Npais, Nfilhos, Nsob, Nger, epson0,
              tau1, tau2, sigma0_low, sigma0_high,
              clip_val, patience, prec, pmut,
              T_da, alpha_da) in enumerate(combinations, start=1):

        tau1_real = tau1 if tau1 is not None else 1.0 / np.sqrt(2 * Nd)
        tau2_real = tau2 if tau2 is not None else 1.0 / np.sqrt(2 * np.sqrt(Nd))

        if Nsob > Nind:
            print(f"[{idx}/{total}] Pulando: Nsob={Nsob} > Nind={Nind}")
            continue
        if Npais > min(Nind, Nsob):
            print(f"[{idx}/{total}] Pulando: Npais={Npais} > min(Nind,Nsob)={min(Nind,Nsob)}")
            continue

        comb_key = (Nind, Npais, Nfilhos, Nsob, Nger, epson0,
                    tau1, tau2, sigma0_low, sigma0_high,
                    clip_val, patience, prec, pmut, T_da, alpha_da)
        if comb_key in tested:
            print(f"[{idx}/{total}] Pulando combinação já calculada.")
            continue

        elapsed_grid = time.perf_counter() - t_grid_start
        eta_str = (str(datetime.timedelta(
                        seconds=int(elapsed_grid / (idx-1) * (total-idx+1))))
                   if idx > 1 else "calculando...")

        print(f"\n{'='*50}")
        print(f"Combinação {idx}/{total}  |  ETA: {eta_str}")
        print(f"Nind={Nind}, Npais={Npais}, Nfilhos={Nfilhos}, Nsob={Nsob}, Nger={Nger}")
        print(f"T_da={T_da}, alpha_da={alpha_da}, prec={prec}, pmut={pmut}, patience={patience}")

        metrics = run_100_execucoes_hibrido(
            data_vectors=data_vectors, NC=NC, J_ref=J_global_ref,
            Nind=Nind, Npais=Npais, Nfilhos=Nfilhos, Nsob=Nsob,
            Nger=Nger, epson0=epson0, tau1=tau1, tau2=tau2,
            sigma0_low=sigma0_low, sigma0_high=sigma0_high,
            clip_val=clip_val, patience=patience,
            prec=prec, pmut=pmut,
            T_da=T_da, alpha_da=alpha_da,
            tol=tol, Nexec=Nexec, base_seed=0
        )

        print(f"  -> Medindo tempo ({N_rep} repetições)...")
        t_runs = []
        for _ in range(N_rep):
            t0 = time.perf_counter()
            run_hibrido(
                data_vectors=data_vectors, NC=NC,
                Nind=Nind, Npais=Npais, Nfilhos=Nfilhos, Nsob=Nsob,
                Nger=Nger, epson0=epson0, tau1=tau1, tau2=tau2,
                sigma0_low=sigma0_low, sigma0_high=sigma0_high,
                clip_val=clip_val, patience=patience,
                prec=prec, pmut=pmut,
                T_da=T_da, alpha_da=alpha_da,
                tol=tol, J_ref=J_global_ref, init_seed=0
            )
            t_runs.append(time.perf_counter() - t0)
        tempo_medio = float(np.mean(t_runs))
        tempo_std   = float(np.std(t_runs))
        print(f"  -> Tempo médio: {tempo_medio:.3f} s | Std: {tempo_std:.3f} s")

        combo_tag = (
            f"Nind{Nind}_Np{Npais}_Nfil{Nfilhos}_Nsob{Nsob}"
            f"_Nger{Nger}_Tda{T_da}_ada{alpha_da}"
        ).replace(".", "p")

        history_D_path     = os.path.join(local_path, f"hib_history_D_{combo_tag}.npy")
        history_Dbest_path = os.path.join(local_path, f"hib_history_Dbest_{combo_tag}.npy")
        history_time_path  = os.path.join(local_path, f"hib_history_time_{combo_tag}.npy")
        np.save(history_D_path,     metrics["best_hist_D"])
        np.save(history_Dbest_path, metrics["best_hist_Dbest"])
        np.save(history_time_path,  metrics["best_hist_time"])

        row = {
            "timestamp":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Nind": Nind, "Npais": Npais, "Nfilhos": Nfilhos,
            "Nsob": Nsob, "Nger": Nger, "epson0": epson0,
            "tau1": tau1, "tau2": tau2,
            "tau1_real": tau1_real, "tau2_real": tau2_real,
            "sigma0_low": sigma0_low, "sigma0_high": sigma0_high,
            "clip_val": clip_val, "patience": patience,
            "prec": prec, "pmut": pmut,
            "T_da": T_da, "alpha_da": alpha_da,
            "tol": tol, "Nexec": Nexec,
            "SR":    metrics["SR"],
            "MBF":   metrics["MBF"],
            "AES":   metrics["AES"],
            "J_best":metrics["J_best"],
            "J_global_reference": float(J_global_ref),
            "tempo_medio_s": tempo_medio,
            "tempo_std_s":   tempo_std,
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
            best_record = {**row,
                "best_hist_D":     metrics["best_hist_D"],
                "best_hist_Dbest": metrics["best_hist_Dbest"],
                "best_hist_time":  metrics["best_hist_time"],
            }
            np.savez(best_record_path, **{k: v for k, v in best_record.items()
                     if not isinstance(v, str)})
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
# 7. Plot Híbrido
# =========================================================
def plot_best_solution_hibrido(data_vectors, cluster_centers, best_record):

    history_D     = np.asarray(best_record["best_hist_D"])
    history_Dbest = np.asarray(best_record["best_hist_Dbest"])
    history_time  = np.asarray(best_record["best_hist_time"])
    J_ref         = J_hard(cluster_centers, data_vectors)

    plt.figure(figsize=(10, 6))
    plt.plot(history_D,     'r-', label='Custo médio da geração')
    plt.plot(history_Dbest, 'k-', label='Melhor custo acumulado (J_best)')
    plt.axhline(y=J_ref, color='b', linestyle='--', linewidth=1.0,
                label='Custo com centros verdadeiros')
    plt.grid(); plt.xlabel('Geração'); plt.ylabel('Custo')
    plt.title('Híbrido ES+DA — convergência por geração')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(local_path, "hibrido_convergencia_geracao.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history_time, history_D,     'r-', label='Custo médio da geração')
    plt.plot(history_time, history_Dbest, 'k-', label='Melhor custo acumulado (J_best)')
    plt.axhline(y=J_ref, color='b', linestyle='--', linewidth=1.0,
                label='Custo com centros verdadeiros')
    plt.grid(); plt.xlabel('Tempo (s)'); plt.ylabel('Custo')
    plt.title('Híbrido ES+DA — convergência por tempo')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(local_path, "hibrido_convergencia_tempo.png"), dpi=150)
    plt.show()

# =========================================================
# 8. Execução principal
# =========================================================
if __name__ == "__main__":

    NC_number = 4   # ← mude aqui para 4, 8, 16, etc.

    # Diretório de saída automático baseado no NC_number
    local_path = os.path.expanduser(f"~/cpe723_hibrido_nc{NC_number}")
    os.makedirs(local_path, exist_ok=True)
    csv_path_hib         = os.path.join(local_path, "grid_search_hibrido_results.csv")
    best_record_path_hib = os.path.join(local_path, "best_record_hibrido.npz")
    print(f"Resultados salvos em: {local_path}")
    data_vectors, cluster_centers = generate_data_r3(P=100, NC=NC_number, sigma=0.1, seed=1)
    J_ref = J_hard(cluster_centers, data_vectors)
    print(f"Custo com centros verdadeiros: {J_ref:.6f}")

    N_rep_timing = 10
    Nexec        = 100

    # ---------------------------------------------------------
    # Fase 1: parâmetros ES fixos no melhor encontrado,
    #         varia T_da e alpha_da para calibrar o passo DA
    # ---------------------------------------------------------
    df_results, best_record = grid_search_hibrido(
        data_vectors=data_vectors,
        cluster_centers=cluster_centers,
        NC=NC_number,
        Nind_values        = [100],
        Npais_values       = [100],
        Nfilhos_values     = [700],
        Nsob_values        = [100],
        Nger_values        = [800],
        epson0_values      = [1e-8],
        tau1_values        = [None],
        tau2_values        = [None],
        sigma0_low_values  = [0.1],
        sigma0_high_values = [0.5],
        clip_val_values    = [None],
        patience_values    = [50],
        prec_values        = [1.0],
        pmut_values        = [1.0],
        T_da_values        = [0.1, 0.5, 1.0, 2.0],   # temperatura do passo DA
        alpha_da_values    = [0.3, 0.5, 0.7, 1.0],   # intensidade do guiamento
        tol=1e-2,
        Nexec=Nexec,
        N_rep=N_rep_timing,
        csv_path=csv_path_hib,
        best_record_path=best_record_path_hib,
    )

    print("\n===== Top 10 combinações (por SR) =====")
    cols_show = ["Nind","Npais","Nfilhos","Nsob","T_da","alpha_da",
                 "SR","MBF","AES","J_best","tempo_medio_s"]
    cols_ok = [c for c in cols_show if c in df_results.columns]
    print(df_results[cols_ok].head(10).to_string())

    if best_record is not None:
        print("\n===== Melhor combinação =====")
        for k in ["Nind","Npais","Nfilhos","Nsob","Nger",
                  "T_da","alpha_da","SR","MBF","AES","J_best"]:
            if k in best_record:
                print(f"  {k} = {best_record[k]}")
        plot_best_solution_hibrido(data_vectors, cluster_centers, best_record)