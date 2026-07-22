"""
Localiza, para cada região (Norte, Nordeste, Sudeste, Sul), as estações
candidatas mais adequadas para representar cada zona batimétrica:

    - Rasa:       H_max próximo de   80 m
    - Plataforma: H_max próximo de  200 m
    - ZEE:        H_max próximo de 2000 m

Para cada zona, o script primeiro filtra as estações cuja profundidade
máxima esteja "razoavelmente perto" do alvo, e depois RANQUEIA essas
candidatas pela COMPLETUDE REAL dos dados — ou seja, quanto da estação
já vem pronta do BNDO (sem precisar estimar por fórmula empírica) para
Temperatura, Salinidade, Densidade, Condutividade e Velocidade do Som.

Isso reaproveita as mesmas funções de processar_estacoes_ctd.py
(completar_*, colapsar_duplicatas_por_profundidade, etc.), garantindo
que a métrica de completude seja calculada exatamente da mesma forma
que já usamos no pipeline principal.

Requisitos: pandas, seawater
    pip install pandas seawater --break-system-packages

Uso:
    python3 explorar_estacoes_por_zona.py
"""

import pandas as pd
from pathlib import Path

import processar_estacoes_ctd as ctd  # reaproveita as funções já validadas

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

ARQUIVOS = {
    "Norte": "../dados_ctd_bndo/Costa Norte 2025 Antares",
    "Nordeste": "../dados_ctd_bndo/Nordeste Cruzeiro do Sul",
    "Sudeste": "../dados_ctd_bndo/Oceano Sudeste",
    "Sul": "../dados_ctd_bndo/Oceano Sul V",
}

# Profundidades-alvo de cada zona batimétrica (conforme a transcrição da aula)
ZONAS_ALVO_M = {
    "Rasa (~80m)": 80,
    "Plataforma (~200m)": 200,
    "ZEE (~2000m)": 2000,
}

# Tolerância para considerar uma estação "candidata" de uma zona: aceita
# estações com H_max entre alvo/FATOR_TOLERANCIA e alvo*FATOR_TOLERANCIA
FATOR_TOLERANCIA = 1.6

# Quantas candidatas (mais próximas do alvo) avaliar em detalhe por zona.
# Avaliar em detalhe é mais caro (roda o pipeline completo de
# completar_*), por isso limitamos a um número razoável.
N_CANDIDATAS_DETALHAR = 6

# Número mínimo de pontos que uma estação precisa ter, DEPOIS do
# pipeline completo, para ser considerada uma candidata válida. Sem
# isso, uma coincidência de lat/lon/data com apenas 1 ou 2 linhas pode
# aparecer com "100% de completude" (trivial, sem significado real) e
# subir artificialmente no ranking.
MIN_PONTOS_ESTACAO_VALIDA = 15

# % mínima de presença de Temperatura e Salinidade para uma candidata
# ser considerada. IMPORTANTE: T e S nunca são estimadas por fórmula
# (são a entrada bruta usada para calcular tudo o mais) — então uma
# estação com T/S incompletos é MUITO pior que uma com c/rho/sigma
# incompletos, já que estes últimos sempre podem ser calculados a
# partir de T/S/P, mas T/S ausentes não têm como ser recuperados.
MIN_PCT_TS_PRESENTE = 90.0

CASAS_DECIMAIS_COORD = 2

# ---------------------------------------------------------------------------
# ETAPA 1 — identificação rápida das estações de cada arquivo
# ---------------------------------------------------------------------------


def identificar_estacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa as linhas do arquivo completo por estação (lat/lon/data),
    calculando profundidade máxima e número de pontos. Rápido, usado só
    para triagem inicial das candidatas."""
    df = df.copy()
    df["_data"] = pd.to_datetime(df["Data-Hora"], errors="coerce").dt.date
    df["_lat_r"] = df["Latitude [deg]"].round(CASAS_DECIMAIS_COORD)
    df["_lon_r"] = df["Longitude [deg]"].round(CASAS_DECIMAIS_COORD)

    agrupado = (
        df.groupby(["_lat_r", "_lon_r", "_data"])
        .agg(
            n_pontos=("Profundidade [m]", "count"),
            profundidade_max=("Profundidade [m]", "max"),
        )
        .reset_index()
        .rename(columns={"_lat_r": "Latitude", "_lon_r": "Longitude", "_data": "Data"})
    )
    return agrupado


# ---------------------------------------------------------------------------
# ETAPA 2 — avaliação detalhada de completude (roda o pipeline completo)
# ---------------------------------------------------------------------------


def avaliar_completude(df_arquivo: pd.DataFrame, lat: float, lon: float,
                        data_str: str) -> dict:
    """
    Roda o pipeline completo (mesmo usado no script principal) para uma
    estação específica, e retorna métricas de completude: quantos % dos
    pontos de cada variável vieram do BNDO (não estimados), mais uma
    pontuação geral combinando as 4 variáveis derivadas.
    """
    estacao = ctd.extrair_estacao(df_arquivo, lat, lon, data_str)
    if estacao.empty:
        return None

    estacao = ctd.completar_profundidade(estacao, lat)
    estacao = estacao.dropna(subset=["Profundidade [m]"]).reset_index(drop=True)
    if estacao.empty:
        return None

    estacao = ctd.colapsar_duplicatas_por_profundidade(estacao)
    estacao = ctd.anotar_qualidade(estacao)
    estacao = ctd.completar_velocidade_som(estacao)
    estacao = ctd.completar_pressao(estacao, lat)
    estacao = ctd.completar_densidade(estacao, lat)
    estacao = ctd.completar_densidade_potencial(estacao)
    estacao = ctd.completar_condutividade(estacao, lat)

    n = len(estacao)
    if n == 0:
        return None

    def pct_bndo(col_origem):
        if col_origem not in estacao.columns:
            return 0.0
        return 100.0 * (estacao[col_origem] == "BNDO").sum() / n

    pct_c = pct_bndo("origem_c")
    pct_rho = pct_bndo("origem_rho")
    pct_sigma = pct_bndo("origem_sigma")
    # Temperatura e Salinidade nunca são estimadas — completude = % de
    # linhas com valor presente (não ausente).
    pct_temp = 100.0 * estacao["Temperatura [°c]"].notna().sum() / n
    pct_sal = 100.0 * estacao["Salinidade [psu]"].notna().sum() / n

    # T/S pesam mais que o dobro dos outros três juntos: sem T/S, nada
    # mais pode ser calculado; com T/S completos, c/rho/sigma sempre
    # podem ser recuperados por fórmula, então sua ausência é um
    # problema bem menor.
    pontuacao_geral = (
        0.35 * pct_temp + 0.35 * pct_sal
        + 0.10 * pct_c + 0.10 * pct_rho + 0.10 * pct_sigma
    )

    return {
        "n_pontos": n,
        "profundidade_max": estacao["Profundidade [m]"].max(),
        "%T_presente": round(pct_temp, 1),
        "%S_presente": round(pct_sal, 1),
        "%c_do_BNDO": round(pct_c, 1),
        "%rho_do_BNDO": round(pct_rho, 1),
        "%sigma_do_BNDO": round(pct_sigma, 1),
        "completude_geral": round(pontuacao_geral, 1),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    for regiao, caminho in ARQUIVOS.items():
        caminho_path = Path(caminho)
        print("=" * 78)
        print(f"REGIÃO: {regiao}  ({caminho})")
        print("=" * 78)

        if not caminho_path.exists():
            print(f"  [ERRO] Arquivo não encontrado: {caminho_path.resolve()}")
            continue

        df = ctd.carregar_arquivo(caminho)
        estacoes = identificar_estacoes(df)

        for nome_zona, alvo_m in ZONAS_ALVO_M.items():
            print(f"\n  --- Zona: {nome_zona} ---")

            limite_inferior = alvo_m / FATOR_TOLERANCIA
            limite_superior = alvo_m * FATOR_TOLERANCIA

            candidatas = estacoes[
                (estacoes["profundidade_max"] >= limite_inferior)
                & (estacoes["profundidade_max"] <= limite_superior)
            ].copy()

            if candidatas.empty:
                print(f"    [AVISO] Nenhuma estação encontrada entre "
                      f"{limite_inferior:.0f}m e {limite_superior:.0f}m "
                      f"de profundidade máxima nesta região.")
                print(f"    Estação mais próxima disponível (qualquer profundidade): ", end="")
                if not estacoes.empty:
                    mais_proxima = estacoes.iloc[
                        (estacoes["profundidade_max"] - alvo_m).abs().argsort().iloc[0]
                    ]
                    print(f"{mais_proxima['profundidade_max']:.0f}m "
                          f"(lat={mais_proxima['Latitude']}, lon={mais_proxima['Longitude']}, "
                          f"data={mais_proxima['Data']})")
                else:
                    print("nenhuma estação no arquivo.")
                continue

            candidatas["dist_alvo"] = (candidatas["profundidade_max"] - alvo_m).abs()
            candidatas = candidatas.sort_values("dist_alvo").head(N_CANDIDATAS_DETALHAR)

            resultados = []
            for _, row in candidatas.iterrows():
                metricas = avaliar_completude(
                    df, row["Latitude"], row["Longitude"], str(row["Data"])
                )
                if metricas is None:
                    continue
                resultados.append({
                    "Latitude": row["Latitude"],
                    "Longitude": row["Longitude"],
                    "Data": row["Data"],
                    **metricas,
                })

            if not resultados:
                print("    [AVISO] Candidatas encontradas, mas nenhuma pôde ser processada.")
                continue

            tabela = pd.DataFrame(resultados)

            descartadas = tabela[tabela["n_pontos"] < MIN_PONTOS_ESTACAO_VALIDA]
            if not descartadas.empty:
                print(f"    [AVISO] {len(descartadas)} candidata(s) descartada(s) por ter(em) "
                      f"menos de {MIN_PONTOS_ESTACAO_VALIDA} pontos (provável coincidência "
                      f"espúria de lat/lon/data, não uma estação real):")
                print(descartadas[["Latitude", "Longitude", "Data", "n_pontos",
                                    "profundidade_max", "completude_geral"]].to_string(index=False))

            tabela = tabela[tabela["n_pontos"] >= MIN_PONTOS_ESTACAO_VALIDA]
            if tabela.empty:
                print("    [AVISO] Nenhuma candidata restante após filtro de tamanho mínimo.")
                continue

            descartadas_ts = tabela[
                (tabela["%T_presente"] < MIN_PCT_TS_PRESENTE)
                | (tabela["%S_presente"] < MIN_PCT_TS_PRESENTE)
            ]
            if not descartadas_ts.empty:
                print(f"    [AVISO] {len(descartadas_ts)} candidata(s) descartada(s) por "
                      f"Temperatura/Salinidade muito incompletos (< {MIN_PCT_TS_PRESENTE:.0f}%) "
                      f"— sem T/S não há como calcular c/rho/sigma nesses pontos:")
                print(descartadas_ts[["Latitude", "Longitude", "Data", "n_pontos",
                                       "%T_presente", "%S_presente"]].to_string(index=False))

            tabela_valida_ts = tabela[
                (tabela["%T_presente"] >= MIN_PCT_TS_PRESENTE)
                & (tabela["%S_presente"] >= MIN_PCT_TS_PRESENTE)
            ]
            if not tabela_valida_ts.empty:
                tabela = tabela_valida_ts
            else:
                print(f"    [AVISO] Nenhuma candidata com T/S >= {MIN_PCT_TS_PRESENTE:.0f}%; "
                      f"mantendo todas as candidatas mesmo assim (revisar manualmente).")

            tabela = tabela.sort_values("completude_geral", ascending=False)
            print(tabela.to_string(index=False))
            melhor = tabela.iloc[0]
            print(f"    >>> Melhor candidata por completude: lat={melhor['Latitude']}, "
                  f"lon={melhor['Longitude']}, data={melhor['Data']}, "
                  f"H_max={melhor['profundidade_max']:.0f}m, "
                  f"completude geral={melhor['completude_geral']:.1f}%")

        print()


if __name__ == "__main__":
    main()