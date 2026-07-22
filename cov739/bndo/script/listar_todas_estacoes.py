"""
Lista TODAS as estações de um arquivo de CTD, sem nenhum filtro de zona
ou tolerância, ordenadas por profundidade máxima crescente.

Objetivo: permitir uma conferência manual direta — você vê a lista
completa de profundidades máximas disponíveis naquele arquivo e pode
confirmar (ou refutar) se existe algum "buraco" na cobertura de
profundidades (por exemplo, nenhuma estação entre 90m e 500m).

Uso:
    python3 listar_todas_estacoes.py "../dados_ctd_bndo/Oceano Sul V"
"""

import sys
import pandas as pd
from pathlib import Path

CASAS_DECIMAIS_COORD = 2


def carregar_arquivo(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(
        caminho, sep=";", na_values=["None", "none", ""],
        encoding="utf-8", engine="python", on_bad_lines="warn",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 listar_todas_estacoes.py <caminho_do_arquivo>")
        sys.exit(1)

    caminho = sys.argv[1]
    if not Path(caminho).exists():
        print(f"[ERRO] Arquivo não encontrado: {Path(caminho).resolve()}")
        sys.exit(1)

    print(f"Carregando {caminho} ...")
    df = carregar_arquivo(caminho)
    print(f"Total de linhas no arquivo: {len(df)}\n")

    df["_data"] = pd.to_datetime(df["Data-Hora"], errors="coerce").dt.date
    df["_lat_r"] = df["Latitude [deg]"].round(CASAS_DECIMAIS_COORD)
    df["_lon_r"] = df["Longitude [deg]"].round(CASAS_DECIMAIS_COORD)

    estacoes = (
        df.groupby(["_lat_r", "_lon_r", "_data"])
        .agg(
            n_pontos=("Profundidade [m]", "count"),
            profundidade_min=("Profundidade [m]", "min"),
            profundidade_max=("Profundidade [m]", "max"),
        )
        .reset_index()
        .rename(columns={"_lat_r": "Latitude", "_lon_r": "Longitude", "_data": "Data"})
        .sort_values("profundidade_max")
        .reset_index(drop=True)
    )

    print(f"Total de estações distintas identificadas: {len(estacoes)}\n")
    pd.set_option("display.max_rows", None)
    print(estacoes.to_string(index=True))

    print("\n--- Resumo da distribuição de profundidades máximas ---")
    print(estacoes["profundidade_max"].describe())

    # Aponta o maior "buraco" (maior salto) na sequência ordenada de
    # profundidades máximas, o que ajuda a visualizar rapidamente onde
    # está a lacuna de cobertura, se houver.
    diffs = estacoes["profundidade_max"].diff()
    idx_maior_salto = diffs.idxmax()
    if pd.notna(idx_maior_salto) and idx_maior_salto > 0:
        prof_antes = estacoes.loc[idx_maior_salto - 1, "profundidade_max"]
        prof_depois = estacoes.loc[idx_maior_salto, "profundidade_max"]
        print(f"\nMaior lacuna na cobertura de profundidades: entre "
              f"{prof_antes:.1f}m e {prof_depois:.1f}m "
              f"(salto de {prof_depois - prof_antes:.1f}m sem nenhuma estação no meio)")


if __name__ == "__main__":
    main()
