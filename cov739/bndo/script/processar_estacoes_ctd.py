"""
Processa estações de CTD do BNDO para as 4 regiões da costa brasileira
(Norte, Nordeste, Sudeste, Sul), em até 3 zonas batimétricas por região
(Rasa ~80m, Plataforma ~200m, ZEE ~2000m — conforme explicado em aula),
e gera:

  1) Um CSV limpo por (região, zona).
  2) Figuras "por região": para cada região, um painel por parâmetro
     (Pressão, Temperatura, Salinidade, Densidade Potencial, Densidade
     in situ, Condutividade, Velocidade do Som), comparando as zonas
     disponíveis daquela região.
  3) Figuras "por zona": para cada zona, o mesmo conjunto de painéis,
     comparando as regiões que têm estação disponível naquela zona.
  4) Cada figura acima em duas versões: linha sólida contínua, e uma
     segunda versão marcando com linha pontilhada os trechos de dado
     ESTIMADO (não vindo diretamente do BNDO).
  5) Uma tabela-resumo (uma linha por região/zona processada).

IMPORTANTE — limitação de dados conhecida:
  A região Sul não possui, na comissão disponível (Oceano Sul V),
  nenhuma estação de água Rasa ou de Plataforma — só estações de
  águas profundas (ZEE). Isso foi confirmado inspecionando todas as 41
  estações do arquivo (ver listar_todas_estacoes.py); a lacuna entre
  ~575m e ~1000m não tem nenhuma estação no meio. Documentado como
  limitação de disponibilidade de dados, conforme a solicitação
  original ao BNDO ("conforme disponibilidade de dados da estação").

IMPORTANTE — decisão pendente sobre QC:
  O filtro de QC (QC_MINIMO_ACEITAVEL) vem DESLIGADO (None) por
  padrão — nada é descartado, apenas os dados são processados como
  vieram. Ajuste essa variável quando decidir o critério com o
  professor/você mesmo.

Requisitos: pandas, matplotlib, seawater
    pip install pandas matplotlib seawater --break-system-packages

Uso:
    python3 processar_estacoes_ctd.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seawater as sw
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

ARQUIVOS = {
    "Norte": "../dados_ctd_bndo/Costa Norte 2025 Antares",
    "Nordeste": "../dados_ctd_bndo/Nordeste Cruzeiro do Sul",
    "Sudeste": "../dados_ctd_bndo/Oceano Sudeste",
    "Sul": "../dados_ctd_bndo/Oceano Sul V",
}

# Estações escolhidas por região e por zona batimétrica, com base na
# completude real dos dados (ver explorar_estacoes_por_zona.py).
# Sul não tem Rasa nem Plataforma disponível nesta comissão (ver nota acima).
ESTACOES_SELECIONADAS = {
    "Norte": {
        "Rasa":       {"lat": 4.80,  "lon": -50.50, "data": "2025-11-05"},  # H~86m
        "Plataforma": {"lat": 3.70,  "lon": -48.61, "data": "2025-10-23"},  # H~304m (n=3020 pontos)
        "ZEE":        {"lat": 7.32,  "lon": -43.87, "data": "2025-10-12"},  # H~2018m
    },
    "Nordeste": {
        "Rasa":       {"lat": -2.23, "lon": -41.28, "data": "2025-04-04"},  # H~50m, T/S 100% completos
        "Plataforma": {"lat": -3.26, "lon": -37.34, "data": "2025-03-30"},  # H~248m
        "ZEE":        {"lat": -2.51, "lon": -38.16, "data": "2025-04-01"},  # H~2065m
    },
    "Sudeste": {
        "Rasa":       {"lat": -23.55, "lon": -43.68, "data": "2002-11-21"}, # H~82m
        "Plataforma": {"lat": -22.62, "lon": -40.43, "data": "2002-11-08"}, # H~214m
        "ZEE":        {"lat": -22.27, "lon": -39.19, "data": "2002-10-23"}, # H~2098m
    },
    "Sul": {
        "ZEE":        {"lat": -30.24, "lon": -37.01, "data": "2018-05-05"}, # H~1925m
        # Rasa e Plataforma: sem estação disponível nesta comissão.
    },
}

CORES_ZONA = {"Rasa": "tab:green", "Plataforma": "tab:orange", "ZEE": "tab:blue"}
CORES_REGIAO = {"Norte": "tab:red", "Nordeste": "tab:orange",
                 "Sudeste": "tab:green", "Sul": "tab:blue"}

CASAS_DECIMAIS_COORD = 2

# Pasta onde salvar os CSVs e os gráficos
PASTA_SAIDA = Path("../resultados_ctd")

# Filtro de QC (ver aviso no topo do arquivo). None = não filtra, só processa.
QC_MINIMO_ACEITAVEL = None

# ---------------------------------------------------------------------------
# FUNÇÕES DE LEITURA E EXTRAÇÃO
# ---------------------------------------------------------------------------


def carregar_arquivo(caminho: str) -> pd.DataFrame:
    """Lê um arquivo CTD do BNDO (CSV separado por ';'), acessando colunas
    por nome (a ordem varia entre arquivos)."""
    df = pd.read_csv(
        caminho,
        sep=";",
        na_values=["None", "none", ""],
        encoding="utf-8",
        engine="python",
        on_bad_lines="warn",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def completar_profundidade(estacao: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """
    Sempre que 'Profundidade [m]' vier vazia mas 'Pressão [db]' estiver
    disponível, calcula a profundidade a partir da pressão (fórmula de
    Saunders & Fofonoff, via seawater.dpth(), que leva em conta a
    latitude). Marca a origem em 'origem_profundidade'.

    IMPORTANTE: esta função deve ser chamada ANTES de
    colapsar_duplicatas_por_profundidade — como o colapso agrupa as
    linhas pela coluna de profundidade, qualquer linha com profundidade
    ausente seria descartada silenciosamente nesse agrupamento, mesmo
    tendo pressão e outros dados válidos.
    """
    estacao = estacao.copy()

    if "Pressão [db]" not in estacao.columns:
        estacao["origem_profundidade"] = "BNDO"
        return estacao

    tem_bndo = estacao["Profundidade [m]"].notna()
    pode_calcular = (~tem_bndo) & estacao["Pressão [db]"].notna()

    estacao["origem_profundidade"] = "ausente"
    estacao.loc[tem_bndo, "origem_profundidade"] = "BNDO"

    if pode_calcular.any():
        estacao.loc[pode_calcular, "Profundidade [m]"] = sw.dpth(
            estacao.loc[pode_calcular, "Pressão [db]"].values, latitude
        )
        estacao.loc[pode_calcular, "origem_profundidade"] = "calculada (dpth/Saunders-Fofonoff)"

    return estacao


def extrair_estacao(df: pd.DataFrame, lat: float, lon: float, data_str: str) -> pd.DataFrame:
    """Filtra o DataFrame completo para manter só as linhas da estação
    (lat/lon/data) especificada, ordenadas por profundidade crescente."""
    df = df.copy()
    df["_data"] = pd.to_datetime(df["Data-Hora"], errors="coerce").dt.date
    df["_lat_r"] = df["Latitude [deg]"].round(CASAS_DECIMAIS_COORD)
    df["_lon_r"] = df["Longitude [deg]"].round(CASAS_DECIMAIS_COORD)

    alvo_data = pd.to_datetime(data_str).date()
    alvo_lat = round(lat, CASAS_DECIMAIS_COORD)
    alvo_lon = round(lon, CASAS_DECIMAIS_COORD)

    mask = (
        (df["_lat_r"] == alvo_lat)
        & (df["_lon_r"] == alvo_lon)
        & (df["_data"] == alvo_data)
    )
    estacao = df[mask].sort_values("Profundidade [m]").reset_index(drop=True)
    estacao = estacao.drop(columns=["_data", "_lat_r", "_lon_r"])
    return estacao


def _combinar_grupo_profundidade(g: pd.DataFrame) -> pd.Series:
    """
    Combina todas as linhas de um mesmo grupo de profundidade (arredondada)
    numa única linha, preservando a COERÊNCIA FÍSICA entre colunas que
    precisam vir do mesmo instante de leitura:

      - Bloco T/S (Temperatura, Salinidade, Temperatura Potencial): pega
        os valores de UMA ÚNICA linha (a primeira que tiver T e S ambos
        preenchidos), nunca combina T de uma linha com S de outra.
      - Bloco P/Condutividade/Densidade/Velocidade do som: pega os
        valores de UMA ÚNICA linha (a que tiver mais desses campos
        preenchidos simultaneamente), pelo mesmo motivo.

    Isso evita o problema de, num grupo com mais de 2 linhas (por
    exemplo, por causa de uma parada do CTD/rosette para coleta de
    amostra de água — comum em dados antigos), acabar misturando
    valores de leituras diferentes e quebrando a relação física entre
    eles (o que produzia o efeito de 'zigue-zague' nos perfis).
    """
    linha = {}

    colunas_ts = [c for c in ["Temperatura [°c]", "Temperatura Potencial [°C]",
                               "QC_Temperatura [Flag]"] if c in g.columns]
    base_ts = [c for c in ["Temperatura [°c]", "Salinidade [psu]"] if c in g.columns]
    if base_ts:
        mask_completo = g[base_ts].notna().all(axis=1)
        candidatos_ts = g[mask_completo] if mask_completo.any() else g
    else:
        candidatos_ts = g
    linha_ts = candidatos_ts.iloc[0]
    for c in colunas_ts + (["Salinidade [psu]"] if "Salinidade [psu]" in g.columns else []) \
            + (["QC_Salinidade [Flag]"] if "QC_Salinidade [Flag]" in g.columns else []):
        linha[c] = linha_ts[c]

    colunas_outras = [c for c in [
        "Pressão [db]", "Condutividade [S/m]", "Velocidade do som [m/s]", "Densidade ro [kg/m³]",
        "QC_Pressão [Flag]", "QC_Condutividade [Flag]", "QC_Velocidade do som [Flag]", "QC_Densidade ro [Flag]",
    ] if c in g.columns]
    base_outras = [c for c in ["Pressão [db]", "Condutividade [S/m]",
                                "Velocidade do som [m/s]", "Densidade ro [kg/m³]"] if c in g.columns]
    if base_outras:
        n_preenchidos = g[base_outras].notna().sum(axis=1)
        linha_outras = g.loc[n_preenchidos.idxmax()]
    else:
        linha_outras = g.iloc[0]
    for c in colunas_outras:
        linha[c] = linha_outras[c]

    if "Data-Hora" in g.columns:
        linha["Data-Hora"] = g["Data-Hora"].iloc[0]

    return pd.Series(linha)


def colapsar_duplicatas_por_profundidade(estacao: pd.DataFrame, casas_decimais_prof: int = 1) -> pd.DataFrame:
    """
    O BNDO frequentemente grava, para uma mesma profundidade nominal, duas
    (ou mais) linhas com timestamps próximos: uma com Pressão/Condutividade/
    Densidade/Velocidade do som, outra com Temperatura/Salinidade — e, em
    alguns casos (paradas de rosette para coleta de amostra), várias linhas
    extras na mesma profundidade nominal.

    Esta função agrupa por profundidade (arredondada) e reconstitui uma
    única linha por profundidade, preservando a coerência física entre
    colunas que precisam vir do mesmo instante de leitura (ver
    _combinar_grupo_profundidade).
    """
    estacao = estacao.copy()
    estacao["Profundidade [m]"] = estacao["Profundidade [m]"].round(casas_decimais_prof)

    colunas_numericas = [
        "Temperatura [°c]", "Salinidade [psu]", "Condutividade [S/m]",
        "Velocidade do som [m/s]", "Densidade ro [kg/m³]",
        "Pressão [db]", "Temperatura Potencial [°C]",
    ]

    colapsada = (
        estacao.groupby("Profundidade [m]", group_keys=True)
        .apply(_combinar_grupo_profundidade, include_groups=False)
        .reset_index()
        .sort_values("Profundidade [m]")
        .reset_index(drop=True)
    )

    # Garante que as colunas numéricas fiquem com NaN "de verdade" (float),
    # e não com o tipo pd.NA (que o matplotlib não sabe interpretar).
    for c in colunas_numericas:
        if c in colapsada.columns:
            colapsada[c] = pd.to_numeric(colapsada[c], errors="coerce")

    return colapsada


def celeridade_mackenzie(temperatura, salinidade, profundidade):
    """
    Equação de Mackenzie (1981) para a velocidade do som na água do mar,
    válida para T entre 2-30°C, S entre 25-40 psu, profundidade até 8000m.

    c = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3
        + 1.340*(S-35) + 1.630e-2*D + 1.675e-7*D^2
        - 1.025e-2*T*(S-35) - 7.139e-13*T*D^3

    onde T = temperatura [°C], S = salinidade [psu], D = profundidade [m].
    """
    T, S, D = temperatura, salinidade, profundidade
    return (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T**2
        + 2.374e-4 * T**3
        + 1.340 * (S - 35)
        + 1.630e-2 * D
        + 1.675e-7 * D**2
        - 1.025e-2 * T * (S - 35)
        - 7.139e-13 * T * D**3
    )


def completar_velocidade_som(estacao: pd.DataFrame) -> pd.DataFrame:
    """
    Sempre que 'Velocidade do som [m/s]' vier vazia do BNDO, calcula o
    valor usando a equação de Mackenzie a partir de T, S e profundidade
    (quando esses três estiverem disponíveis), e marca a origem do dado
    numa nova coluna 'origem_c'.
    """
    estacao = estacao.copy()
    tem_bndo = estacao["Velocidade do som [m/s]"].notna()

    pode_calcular = (
        estacao["Temperatura [°c]"].notna()
        & estacao["Salinidade [psu]"].notna()
        & estacao["Profundidade [m]"].notna()
    )

    estacao["origem_c"] = "ausente"
    estacao.loc[tem_bndo, "origem_c"] = "BNDO"

    calcular_aqui = (~tem_bndo) & pode_calcular
    estacao.loc[calcular_aqui, "Velocidade do som [m/s]"] = celeridade_mackenzie(
        estacao.loc[calcular_aqui, "Temperatura [°c]"],
        estacao.loc[calcular_aqui, "Salinidade [psu]"],
        estacao.loc[calcular_aqui, "Profundidade [m]"],
    )
    estacao.loc[calcular_aqui, "origem_c"] = "Mackenzie (calculado)"

    return estacao


def completar_pressao(estacao: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """
    Sempre que 'Pressão [db]' vier vazia do BNDO, calcula o valor a
    partir da profundidade (fórmula de Saunders & Fofonoff, via
    seawater.pres(), que leva em conta a latitude). Marca a origem em
    'origem_pressao'.

    Deve ser chamada depois de completar_profundidade() (para garantir
    que 'Profundidade [m]' já esteja completa) e antes de
    completar_densidade()/completar_condutividade() (que dependem da
    pressão já preenchida).
    """
    estacao = estacao.copy()
    if "Pressão [db]" not in estacao.columns:
        estacao["Pressão [db]"] = pd.NA

    tem_bndo = estacao["Pressão [db]"].notna()
    pode_calcular = (~tem_bndo) & estacao["Profundidade [m]"].notna()

    estacao["origem_pressao"] = "ausente"
    estacao.loc[tem_bndo, "origem_pressao"] = "BNDO"

    if pode_calcular.any():
        estacao.loc[pode_calcular, "Pressão [db]"] = sw.pres(
            estacao.loc[pode_calcular, "Profundidade [m]"].values, latitude
        )
        estacao.loc[pode_calcular, "origem_pressao"] = "calculada (pres/Saunders-Fofonoff)"

    estacao["Pressão [db]"] = pd.to_numeric(estacao["Pressão [db]"], errors="coerce")
    return estacao


def completar_densidade(estacao: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """
    Sempre que 'Densidade ro [kg/m³]' vier vazia do BNDO, calcula o valor
    pela equação de estado da água do mar EOS-80 (UNESCO 1980), a partir
    de Salinidade, Temperatura e Pressão — via seawater.dens(). Marca a
    origem em 'origem_rho'.

    Pressupõe que 'Pressão [db]' já foi completada por completar_pressao().
    """
    estacao = estacao.copy()
    tem_bndo = estacao["Densidade ro [kg/m³]"].notna()

    pode_calcular = (
        estacao["Temperatura [°c]"].notna()
        & estacao["Salinidade [psu]"].notna()
        & estacao["Pressão [db]"].notna()
    )

    estacao["origem_rho"] = "ausente"
    estacao.loc[tem_bndo, "origem_rho"] = "BNDO"

    calcular_aqui = (~tem_bndo) & pode_calcular
    if calcular_aqui.any():
        estacao.loc[calcular_aqui, "Densidade ro [kg/m³]"] = sw.dens(
            estacao.loc[calcular_aqui, "Salinidade [psu]"].values,
            estacao.loc[calcular_aqui, "Temperatura [°c]"].values,
            estacao.loc[calcular_aqui, "Pressão [db]"].values,
        )
        estacao.loc[calcular_aqui, "origem_rho"] = "EOS-80 (calculado)"

    return estacao


def completar_condutividade(estacao: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """
    Sempre que 'Condutividade [S/m]' vier vazia do BNDO, calcula o valor a
    partir de Salinidade, Temperatura e Pressão, usando a escala prática
    de salinidade PSS-78 (via seawater.cndr(), que retorna a razão de
    condutividade R = C(S,T,P)/C(35,15,0)).

    A condutividade de referência C(35,15,0) = 42.914 mS/cm = 4.2914 S/m
    é a constante padrão da escala PSS-78. Marca a origem em 'origem_sigma'.

    Pressupõe que 'Pressão [db]' já foi completada por completar_pressao().
    """
    estacao = estacao.copy()
    C_REFERENCIA_S_POR_M = 4.2914  # 42.914 mS/cm convertido para S/m

    tem_bndo = estacao["Condutividade [S/m]"].notna()

    pode_calcular = (
        estacao["Temperatura [°c]"].notna()
        & estacao["Salinidade [psu]"].notna()
        & estacao["Pressão [db]"].notna()
    )

    estacao["origem_sigma"] = "ausente"
    estacao.loc[tem_bndo, "origem_sigma"] = "BNDO"

    calcular_aqui = (~tem_bndo) & pode_calcular
    if calcular_aqui.any():
        razao = sw.cndr(
            estacao.loc[calcular_aqui, "Salinidade [psu]"].values,
            estacao.loc[calcular_aqui, "Temperatura [°c]"].values,
            estacao.loc[calcular_aqui, "Pressão [db]"].values,
        )
        estacao.loc[calcular_aqui, "Condutividade [S/m]"] = razao * C_REFERENCIA_S_POR_M
        estacao.loc[calcular_aqui, "origem_sigma"] = "PSS-78 (calculado)"

    return estacao


def completar_densidade_potencial(estacao: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a densidade potencial (ρθ) — a densidade que a parcela de
    água teria se fosse trazida adiabaticamente até a superfície (pressão
    de referência = 0), removendo o efeito de compressão da pressão in
    situ. Usa seawater.pden(S, T, P, pr=0), que já calcula internamente a
    temperatura potencial necessária.

    Diferente dos outros 'completar_*', aqui não há valor do BNDO para
    comparar — o BNDO não fornece densidade potencial diretamente (só
    'Temperatura Potencial' aparece nalguns arquivos) — então o cálculo
    é sempre feito quando T, S e P estiverem disponíveis.
    """
    estacao = estacao.copy()
    estacao["Densidade Potencial [kg/m³]"] = pd.NA

    pode_calcular = (
        estacao["Temperatura [°c]"].notna()
        & estacao["Salinidade [psu]"].notna()
        & estacao["Pressão [db]"].notna()
    )

    estacao["origem_rho_potencial"] = "ausente"
    if pode_calcular.any():
        estacao.loc[pode_calcular, "Densidade Potencial [kg/m³]"] = sw.pden(
            estacao.loc[pode_calcular, "Salinidade [psu]"].values,
            estacao.loc[pode_calcular, "Temperatura [°c]"].values,
            estacao.loc[pode_calcular, "Pressão [db]"].values,
            pr=0,
        )
        estacao.loc[pode_calcular, "origem_rho_potencial"] = "EOS-80, pr=0 (calculado)"

    estacao["Densidade Potencial [kg/m³]"] = pd.to_numeric(
        estacao["Densidade Potencial [kg/m³]"], errors="coerce"
    )
    return estacao


def extrair_digito_qc(flag) -> "int | None":
    """
    Extrai o dígito de qualidade principal de uma flag de QC.

    As flags observadas nos dados vêm em duas formas:
      - Um único dígito (ex: '9'), usado tipicamente quando o valor está
        ausente ('9' = sem valor).
      - Um código composto de 3 dígitos (ex: '202'), onde o PRIMEIRO
        dígito é o QC geral do BNDO (na mesma escala 0-9: 2=correto,
        3=inconsistente, 4=duvidoso, etc. — ver arquivo Leia-me do BNDO).

    Retorna o primeiro dígito como inteiro, ou None se não for possível
    interpretar (valor ausente/malformado).
    """
    if pd.isna(flag):
        return None
    s = str(flag).strip()
    if not s:
        return None
    try:
        return int(s[0])
    except ValueError:
        return None


def anotar_qualidade(estacao: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas com o dígito de qualidade interpretado para
    Temperatura, Salinidade e Velocidade do som, para inspeção/filtragem."""
    estacao = estacao.copy()
    mapeamento = {
        "QC_Temperatura [Flag]": "qc_temperatura",
        "QC_Salinidade [Flag]": "qc_salinidade",
        "QC_Velocidade do som [Flag]": "qc_velocidade_som",
    }
    for col_original, col_nova in mapeamento.items():
        if col_original in estacao.columns:
            estacao[col_nova] = estacao[col_original].apply(extrair_digito_qc)
    return estacao


def aplicar_filtro_qc(estacao: pd.DataFrame, qc_minimo) -> pd.DataFrame:
    """
    Se QC_MINIMO_ACEITAVEL não for None, filtra as linhas mantendo apenas
    aquelas cujo dígito de qualidade (temperatura E salinidade) esteja no
    conjunto aceitável. Caso contrário, retorna o DataFrame sem alteração.
    """
    if qc_minimo is None:
        return estacao

    aceitos = qc_minimo if isinstance(qc_minimo, (list, set, tuple)) else [qc_minimo]

    mask = pd.Series(True, index=estacao.index)
    for col in ("qc_temperatura", "qc_salinidade"):
        if col in estacao.columns:
            mask &= estacao[col].isin(aceitos)

    return estacao[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# PIPELINE COMPLETO PARA UMA ESTAÇÃO (região + zona)
# ---------------------------------------------------------------------------


def processar_estacao_completa(df_arquivo: pd.DataFrame, lat: float, lon: float,
                                data_str: str, rotulo_log: str = "") -> "pd.DataFrame | None":
    """
    Roda o pipeline completo (extração, colapso de duplicatas, e todos os
    completar_*) para uma estação específica, e imprime um resumo de uma
    linha com a % de dados vindos do BNDO vs. estimados.
    """
    estacao = extrair_estacao(df_arquivo, lat, lon, data_str)
    if estacao.empty:
        print(f"  [AVISO] {rotulo_log}: nenhum ponto encontrado para "
              f"lat={lat}, lon={lon}, data={data_str}")
        return None

    n_antes = len(estacao)
    estacao = completar_profundidade(estacao, lat)
    estacao = estacao.dropna(subset=["Profundidade [m]"]).reset_index(drop=True)
    if estacao.empty:
        print(f"  [AVISO] {rotulo_log}: todas as linhas descartadas por "
              f"falta de profundidade E pressão")
        return None

    # Checagem de sanidade: se alguma profundidade (arredondada) tiver
    # muito mais que ~2-3 linhas associadas, é sinal de que a estação
    # provavelmente mistura MAIS de um lance/mergulho de CTD na mesma
    # posição/data (comum em dados antigos) — o que produz um perfil
    # "embaralhado" (zigue-zague) depois do colapso, já que linhas de
    # lances diferentes acabam competindo pelo mesmo intervalo de
    # profundidade sem nenhuma ordem temporal consistente entre elas.
    contagem_por_profundidade = estacao["Profundidade [m]"].round(1).value_counts()
    max_repeticoes = contagem_por_profundidade.max() if not contagem_por_profundidade.empty else 0
    if max_repeticoes > 2:
        print(f"  [AVISO] {rotulo_log}: até {int(max_repeticoes)} linhas na mesma "
              f"profundidade (esperado ~2) — esta estação provavelmente mistura "
              f"múltiplos lances de CTD. O perfil resultante pode ficar com "
              f"'ruído' (zigue-zague). Considere trocar por outra candidata.")

    estacao = colapsar_duplicatas_por_profundidade(estacao)
    estacao = anotar_qualidade(estacao)
    estacao = completar_velocidade_som(estacao)
    estacao = completar_pressao(estacao, lat)
    estacao = completar_densidade(estacao, lat)
    estacao = completar_densidade_potencial(estacao)
    estacao = completar_condutividade(estacao, lat)
    estacao = aplicar_filtro_qc(estacao, QC_MINIMO_ACEITAVEL)

    if estacao.empty:
        print(f"  [AVISO] {rotulo_log}: nenhuma linha restante após filtro de QC")
        return None

    def pct_bndo(col_origem):
        if col_origem not in estacao.columns:
            return 0.0
        return 100.0 * (estacao[col_origem] == "BNDO").sum() / len(estacao)

    print(f"  {rotulo_log}: {n_antes} linhas brutas -> {len(estacao)} pontos "
          f"(H_max={estacao['Profundidade [m]'].max():.1f}m) | "
          f"c={pct_bndo('origem_c'):.0f}% BNDO, "
          f"ρ={pct_bndo('origem_rho'):.0f}% BNDO, "
          f"σ={pct_bndo('origem_sigma'):.0f}% BNDO")

    return estacao


# ---------------------------------------------------------------------------
# PLOTAGEM
# ---------------------------------------------------------------------------


def plotar_com_origem(ax, x: pd.Series, y: pd.Series, origem: "pd.Series | None",
                       color, label: "str | None" = None) -> None:
    """
    Plota uma curva (x, y) marcando com linha SÓLIDA os trechos onde o
    valor veio do BNDO, e com linha PONTILHADA os trechos onde o valor
    foi estimado (calculado por alguma das fórmulas empíricas). Mantém a
    continuidade visual entre os trechos (sem 'buracos' na curva).

    Se 'origem' for None (parâmetro que nunca é estimado, como
    Temperatura e Salinidade), a curva inteira é plotada sólida, como
    antes.
    """
    x = x.to_numpy()
    y = y.to_numpy()

    if origem is None:
        ax.plot(x, y, linestyle="-", color=color, label=label)
        return

    is_estimado = (origem.to_numpy() != "BNDO")
    n = len(x)
    if n == 0:
        return

    primeiro_trecho = True
    inicio = 0
    for i in range(1, n + 1):
        fim_do_trecho = (i == n) or (is_estimado[i] != is_estimado[inicio])
        if fim_do_trecho:
            # Inclui o primeiro ponto do próximo trecho (quando existir)
            # para que as linhas se conectem sem deixar espaço em branco.
            fim_idx = min(i + 1, n)
            estilo = ":" if is_estimado[inicio] else "-"
            ax.plot(x[inicio:fim_idx], y[inicio:fim_idx], linestyle=estilo,
                    color=color, label=label if primeiro_trecho else None)
            primeiro_trecho = False
            inicio = i


# Parâmetros na ordem pedida em aula: pressão, temperatura, salinidade,
# densidade potencial, densidade in situ, condutividade, velocidade do som.
# ---------------------------------------------------------------------------
# COMPARAÇÃO COM A FÓRMULA TEÓRICA (gás ideal x fórmula geral de fluidos)
# ---------------------------------------------------------------------------


def calcular_c_diferencas_finitas(estacao: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a velocidade do som usando DIRETAMENTE a fórmula geral e exata
    para fluidos (vista em aula):

        c = sqrt( (∂P/∂ρ)_S )

    Aqui (∂P/∂ρ) é estimado por DIFERENÇAS FINITAS a partir dos próprios
    dados medidos de Pressão e Densidade in situ do perfil — ou seja, não
    usa nenhuma fórmula empírica (Mackenzie/UNESCO), só a definição física
    pura aplicada aos dados reais. Isso permite comparar a fórmula teórica
    diretamente com o valor empírico (Mackenzie) já calculado.

    Pressão é convertida de dbar para Pa (1 dbar = 10.000 Pa) antes da
    derivada, para que o resultado saia em unidades corretas (m/s).

    Pontos onde a densidade praticamente não varia entre medições
    vizinhas (camada bem misturada) são descartados, pois a divisão por
    Δρ≈0 explode numericamente e não tem significado físico ali.
    """
    estacao = estacao.copy()
    P_pa = estacao["Pressão [db]"].to_numpy() * 1.0e4
    rho = estacao["Densidade ro [kg/m³]"].to_numpy()

    dP = np.gradient(P_pa)
    drho = np.gradient(rho)

    with np.errstate(divide="ignore", invalid="ignore"):
        c2 = dP / drho

    # Descarta pontos com Δρ desprezível (camada bem misturada) ou
    # resultado sem sentido físico (c² negativo, o que indicaria
    # densidade diminuindo com a pressão — instabilidade não realista
    # aqui, e sim ruído numérico)
    invalido = (np.abs(drho) < 1e-3) | (c2 <= 0) | ~np.isfinite(c2)
    c = np.sqrt(np.where(invalido, np.nan, c2))

    estacao["Velocidade do som (dP/dρ) [m/s]"] = c
    return estacao


def calcular_c_gas_ideal(estacao: pd.DataFrame, gamma_agua: float = 1.01) -> pd.DataFrame:
    """
    Calcula a velocidade do som usando a fórmula de GÁS IDEAL:

        c = sqrt( γ P / ρ )

    aplicada (propositalmente, para efeito de comparação didática) aos
    dados de água do mar — mesmo essa fórmula sendo estritamente válida
    apenas para gases ideais, não para líquidos.

    γ (gamma) da água líquida é aproximadamente 1,01 (Cp/Cv muito
    próximo de 1, já que líquidos têm capacidade calorífica quase igual
    a pressão e volume constantes — bem diferente do ar, onde γ≈1,4).

    O objetivo é justamente mostrar que essa fórmula SUBESTIMA
    grosseiramente a velocidade do som na água — porque, num líquido,
    a pressão ambiente não está diretamente ligada à compressibilidade
    do meio (ao contrário de um gás, onde P e ρ são acopladas pela
    equação de estado PV=nRT). O que realmente importa para a
    compressibilidade de um líquido é o módulo de compressibilidade
    (bulk modulus), não a pressão hidrostática em si.
    """
    estacao = estacao.copy()
    P_pa = estacao["Pressão [db]"].to_numpy() * 1.0e4
    rho = estacao["Densidade ro [kg/m³]"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.sqrt(gamma_agua * P_pa / rho)

    estacao["Velocidade do som (gás ideal) [m/s]"] = c
    return estacao


def gerar_figura_comparacao_formula(estacoes_zee: dict, caminho_saida: Path) -> None:
    """
    Gera uma figura com um painel por região, comparando 3 formas de
    obter a velocidade do som ao longo da profundidade (usando a
    estação de ZEE de cada região, por ser a mais profunda e ter maior
    variação de pressão/densidade, o que deixa a comparação mais clara):

      1) Mackenzie/BNDO — valor empírico já usado no resto do trabalho
      2) sqrt(dP/dρ) — fórmula geral e exata, calculada por diferenças
         finitas diretamente dos dados medidos (sem fórmula empírica)
      3) sqrt(γP/ρ) — fórmula de gás ideal, aplicada "errada" de
         propósito à água, para evidenciar por que ela não serve para
         líquidos.
    """
    regioes_com_zee = {r: df for r, df in estacoes_zee.items() if df is not None}
    if not regioes_com_zee:
        print("  [AVISO] Nenhuma estação de ZEE disponível para a comparação de fórmulas.")
        return

    n = len(regioes_com_zee)
    fig, eixos = plt.subplots(1, n, figsize=(6 * n, 8), sharey=True)
    if n == 1:
        eixos = [eixos]

    for ax, (regiao, estacao) in zip(eixos, regioes_com_zee.items()):
        estacao = calcular_c_diferencas_finitas(estacao)
        estacao = calcular_c_gas_ideal(estacao)

        ax.plot(estacao["Velocidade do som [m/s]"], estacao["Profundidade [m]"],
                color="tab:blue", label="Mackenzie / BNDO (empírica)")
        ax.plot(estacao["Velocidade do som (dP/dρ) [m/s]"], estacao["Profundidade [m]"],
                color="tab:green", linestyle="--", label=r"$\sqrt{dP/d\rho}$ (geral, dados reais)")
        ax.plot(estacao["Velocidade do som (gás ideal) [m/s]"], estacao["Profundidade [m]"],
                color="tab:red", linestyle=":", label=r"$\sqrt{\gamma P/\rho}$ (gás ideal — errado p/ líquido)")

        ax.set_xlabel("Velocidade do som [m/s]")
        ax.set_title(f"{regiao} (ZEE)")
        ax.grid(alpha=0.3)

    eixos[0].set_ylabel("Profundidade [m]")
    eixos[0].invert_yaxis()
    eixos[0].legend(fontsize=8, loc="lower right")

    fig.suptitle("Velocidade do Som: Fórmula Empírica x Fórmula Geral (dP/dρ) x Gás Ideal", fontsize=13)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# GRÁFICOS DE DISPERSÃO PROPRIEDADE x PROPRIEDADE
# (c×T, S×T, σ×T, c×S, σ×S — função [k=0] e derivada [k=1])
# ---------------------------------------------------------------------------

# Todos os pares (y, x) entre {T, S, σ (condutividade), c (velocidade)},
# exceto S×S (sem sentido) e σ×c (fora do que foi pedido em aula).
PARAMETROS_DISPERSAO = [
    ("Temperatura [°c]", "Velocidade do som [m/s]", "Temperatura [°C]", "Velocidade do som [m/s]", "c × T"),
    ("Temperatura [°c]", "Salinidade [psu]", "Temperatura [°C]", "Salinidade [psu]", "S × T"),
    ("Temperatura [°c]", "Condutividade [S/m]", "Temperatura [°C]", "Condutividade [S/m]", "σ × T"),
    ("Salinidade [psu]", "Velocidade do som [m/s]", "Salinidade [psu]", "Velocidade do som [m/s]", "c × S"),
    ("Salinidade [psu]", "Condutividade [S/m]", "Salinidade [psu]", "Condutividade [S/m]", "σ × S"),
]


def gerar_figura_dispersao(curvas: dict, caminho_saida: Path, titulo: str) -> None:
    """
    Gera uma figura com 5 painéis de dispersão propriedade x propriedade
    (não profundidade x propriedade, como nas figuras de perfil):
    c×T, S×T, σ×T, c×S, σ×S.

    Cada curva liga os pontos na ordem de profundidade crescente (não é
    um scatter solto), o que revela a trajetória da massa d'água ao
    longo da coluna — o clássico diagrama usado para identificar massas
    d'água em oceanografia.

    curvas: dict {nome_da_curva: {"dados": DataFrame, "cor": cor}}
    """
    fig, eixos = plt.subplots(2, 3, figsize=(18, 11))
    eixos = eixos.flatten()

    for ax, (col_x, col_y, rotulo_x, rotulo_y, titulo_painel) in zip(eixos, PARAMETROS_DISPERSAO):
        for nome_curva, info in curvas.items():
            df = info["dados"]
            cor = info["cor"]
            if col_x in df.columns and col_y in df.columns:
                ax.plot(df[col_x], df[col_y], color=cor, label=nome_curva,
                        marker=".", markersize=2, linewidth=1)
        ax.set_xlabel(rotulo_x)
        ax.set_ylabel(rotulo_y)
        ax.set_title(titulo_painel)
        ax.grid(alpha=0.3)

    eixos[0].legend(fontsize=8)
    eixos[-1].axis("off")  # 6º espaço sobra vazio (5 painéis de 6 disponíveis)

    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def gerar_figura_dispersao_derivada(curvas: dict, caminho_saida: Path, titulo: str) -> None:
    """
    Versão 'k=1' dos mesmos 5 painéis: em vez do valor da propriedade em
    si, plota a DERIVADA dY/dX (taxa de variação local), calculada por
    diferenças finitas (numpy.gradient) ao longo do perfil.

    Um valor grande de |dY/dX| aponta uma região onde a propriedade X
    influencia fortemente a propriedade Y — por exemplo, um pico em
    |dc/dT| marca onde a temperatura mais afeta a velocidade do som
    naquele perfil (tipicamente perto da termoclina).
    """
    fig, eixos = plt.subplots(2, 3, figsize=(18, 11))
    eixos = eixos.flatten()

    for ax, (col_x, col_y, rotulo_x, rotulo_y, titulo_painel) in zip(eixos, PARAMETROS_DISPERSAO):
        for nome_curva, info in curvas.items():
            df = info["dados"]
            cor = info["cor"]
            if col_x in df.columns and col_y in df.columns:
                x = df[col_x].to_numpy()
                y = df[col_y].to_numpy()
                valido = np.isfinite(x) & np.isfinite(y)
                if valido.sum() < 3:
                    continue
                dx = np.gradient(x[valido])
                dy = np.gradient(y[valido])
                with np.errstate(divide="ignore", invalid="ignore"):
                    derivada = np.where(np.abs(dx) > 1e-6, dy / dx, np.nan)
                ax.plot(x[valido], derivada, color=cor, label=nome_curva,
                        marker=".", markersize=2, linewidth=1)
        ax.axhline(y=0, color="gray", linewidth=0.6, alpha=0.6)
        ax.set_xlabel(rotulo_x)
        ax.set_ylabel(f"d({rotulo_y.split(' ')[0]})/d({rotulo_x.split(' ')[0]})")
        ax.set_title(f"d({titulo_painel.split(' × ')[0]})/d({titulo_painel.split(' × ')[1]})")
        ax.grid(alpha=0.3)

    eixos[0].legend(fontsize=8)
    eixos[-1].axis("off")

    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


PARAMETROS_PERFIL = [
    ("Pressão [db]", "Pressão [db]", "Perfil de Pressão", "origem_pressao"),
    ("Temperatura [°c]", "Temperatura [°C]", "Perfil de Temperatura", None),
    ("Salinidade [psu]", "Salinidade [psu]", "Perfil de Salinidade", None),
    ("Densidade Potencial [kg/m³]", "Densidade potencial [kg/m³]", "Perfil de Densidade Potencial", "origem_rho_potencial"),
    ("Densidade ro [kg/m³]", "Densidade in situ [kg/m³]", "Perfil de Densidade in situ", "origem_rho"),
    ("Condutividade [S/m]", "Condutividade [S/m]", "Perfil de Condutividade", "origem_sigma"),
    ("Velocidade do som [m/s]", "Velocidade do som [m/s]", "Perfil de Velocidade do Som", "origem_c"),
]


def gerar_figura_perfis(curvas: dict, caminho_saida: Path, titulo: str,
                         mostrar_origem: bool = False) -> None:
    """
    Gera uma figura com 7 painéis (Pressão, Temperatura, Salinidade,
    Densidade Potencial, Densidade in situ, Condutividade, Velocidade do
    Som), cada um mostrando todas as curvas fornecidas.

    Parâmetros:
        curvas: dict {nome_da_curva: {"dados": DataFrame, "cor": cor}}
            Cada curva pode representar uma zona (dentro de uma figura
            "por região") ou uma região (dentro de uma figura "por
            zona"). Cada curva para naturalmente na sua própria
            profundidade máxima real — não há mais linha de referência
            teórica fixa; em vez disso, cada curva ganha uma linha
            pontilhada fina na SUA profundidade máxima, na SUA cor.
        caminho_saida: caminho do arquivo .png a ser salvo
        titulo: título principal da figura
        mostrar_origem: se True, cada curva é desenhada com trechos
            sólidos (dado real do BNDO) e pontilhados (dado estimado por
            fórmula empírica).
    """
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 4)
    posicoes = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

    eixos = []
    for (col, rotulo_x, titulo_painel, col_origem), (linha, coluna) in zip(PARAMETROS_PERFIL, posicoes):
        ax = fig.add_subplot(gs[linha, coluna], sharey=eixos[0] if eixos else None)
        for nome_curva, info in curvas.items():
            df = info["dados"]
            cor = info["cor"]
            if col not in df.columns:
                continue
            origem = None
            if mostrar_origem and col_origem is not None and col_origem in df.columns:
                origem = df[col_origem]
            plotar_com_origem(ax, df[col], df["Profundidade [m]"], origem, cor, label=nome_curva)
        ax.set_xlabel(rotulo_x)
        ax.set_title(titulo_painel)
        ax.grid(alpha=0.3)
        eixos.append(ax)

    eixos[0].set_ylabel("Profundidade [m]")
    eixos[0].invert_yaxis()

    # Linha pontilhada fina na profundidade máxima real de cada curva
    # (na cor da própria curva), em todos os painéis.
    profundidades_max = {}
    for nome_curva, info in curvas.items():
        df = info["dados"]
        if "Profundidade [m]" in df.columns and not df["Profundidade [m]"].dropna().empty:
            profundidades_max[nome_curva] = df["Profundidade [m]"].dropna().max()

    for ax in eixos:
        for nome_curva, prof_max in profundidades_max.items():
            ax.axhline(y=prof_max, color=curvas[nome_curva]["cor"],
                       linestyle=":", linewidth=0.8, alpha=0.5)

    # Rótulos com a profundidade máxima de cada curva, só no primeiro painel
    for nome_curva, prof_max in profundidades_max.items():
        cor = curvas[nome_curva]["cor"]
        eixos[0].text(eixos[0].get_xlim()[1], prof_max, f" {nome_curva}: {prof_max:.0f}m",
                      color=cor, fontsize=7, va="center", ha="left")

    eixos[0].legend(fontsize=8)

    if mostrar_origem:
        fig.text(0.5, 0.01,
                  "Linha sólida = dado medido (BNDO)     |     Linha pontilhada = dado estimado (fórmula empírica)",
                  ha="center", fontsize=10, style="italic")

    # 8º espaço sobra vazio (7 painéis usados de 8 disponíveis)
    fig.add_subplot(gs[1, 3]).axis("off")

    fig.suptitle(titulo, fontsize=14)
    if mostrar_origem:
        fig.tight_layout(rect=[0, 0.03, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    estacoes_processadas = {regiao: {} for regiao in ARQUIVOS}

    for regiao, caminho in ARQUIVOS.items():
        print(f"\n=== Região: {regiao} ===")
        caminho_path = Path(caminho)
        if not caminho_path.exists():
            print(f"  [ERRO] Arquivo não encontrado: {caminho_path.resolve()}")
            continue

        df = carregar_arquivo(caminho)
        zonas_da_regiao = ESTACOES_SELECIONADAS.get(regiao, {})

        if not zonas_da_regiao:
            print(f"  [AVISO] Nenhuma zona configurada para {regiao} em ESTACOES_SELECIONADAS")
            continue

        for zona, info in zonas_da_regiao.items():
            estacao = processar_estacao_completa(
                df, info["lat"], info["lon"], info["data"],
                rotulo_log=f"{regiao} / {zona}",
            )
            if estacao is None:
                continue

            estacoes_processadas[regiao][zona] = estacao
            caminho_csv = PASTA_SAIDA / f"estacao_{regiao}_{zona}.csv"
            estacao.to_csv(caminho_csv, sep=";", index=False)

    if not any(estacoes_processadas.values()):
        print("\n[ERRO] Nenhuma estação foi processada com sucesso. "
              "Confira os caminhos e as coordenadas em ESTACOES_SELECIONADAS.")
        return

    # -----------------------------------------------------------------
    # Figuras "por região": zonas como curvas dentro de cada região
    # -----------------------------------------------------------------
    print("\nGerando figuras por região...")
    for regiao, zonas in estacoes_processadas.items():
        if not zonas:
            continue

        curvas = {zona: {"dados": df_z, "cor": CORES_ZONA.get(zona, "black")}
                  for zona, df_z in zonas.items()}

        nota = ""
        if len(zonas) < 3:
            faltando = [z for z in ["Rasa", "Plataforma", "ZEE"] if z not in zonas]
            nota = f" (sem dados de: {', '.join(faltando)})"

        caminho1 = PASTA_SAIDA / f"perfis_regiao_{regiao}.png"
        gerar_figura_perfis(
            curvas, caminho1,
            titulo=f"Perfis CTD — Região {regiao}{nota}",
            mostrar_origem=False,
        )

        caminho2 = PASTA_SAIDA / f"perfis_regiao_{regiao}_com_origem.png"
        gerar_figura_perfis(
            curvas, caminho2,
            titulo=f"Perfis CTD — Região {regiao} — Origem dos Dados{nota}",
            mostrar_origem=True,
        )
        print(f"  {regiao}: {caminho1.name}, {caminho2.name}")

    # -----------------------------------------------------------------
    # Figuras "por zona": regiões como curvas dentro de cada zona
    # -----------------------------------------------------------------
    print("\nGerando figuras por zona...")
    for zona in ["Rasa", "Plataforma", "ZEE"]:
        curvas = {}
        for regiao, zonas in estacoes_processadas.items():
            if zona in zonas:
                curvas[regiao] = {"dados": zonas[zona], "cor": CORES_REGIAO.get(regiao, "black")}

        if not curvas:
            print(f"  [AVISO] Nenhuma região tem estação para a zona '{zona}', pulando.")
            continue

        nota = ""
        regioes_faltando = [r for r in ESTACOES_SELECIONADAS if r not in curvas]
        if regioes_faltando:
            nota = f" (sem dados de: {', '.join(regioes_faltando)})"

        caminho1 = PASTA_SAIDA / f"perfis_zona_{zona}.png"
        gerar_figura_perfis(
            curvas, caminho1,
            titulo=f"Perfis CTD — Zona {zona} (Comparação entre Regiões){nota}",
            mostrar_origem=False,
        )

        caminho2 = PASTA_SAIDA / f"perfis_zona_{zona}_com_origem.png"
        gerar_figura_perfis(
            curvas, caminho2,
            titulo=f"Perfis CTD — Zona {zona} — Origem dos Dados{nota}",
            mostrar_origem=True,
        )
        print(f"  {zona}: {caminho1.name}, {caminho2.name}")

    # -----------------------------------------------------------------
    # Comparação com a fórmula teórica (gás ideal x fórmula geral)
    # -----------------------------------------------------------------
    print("\nGerando comparação com a fórmula teórica...")
    estacoes_zee = {
        regiao: zonas.get("ZEE")
        for regiao, zonas in estacoes_processadas.items()
        if "ZEE" in zonas
    }
    caminho_formula = PASTA_SAIDA / "comparacao_formula_teorica.png"
    gerar_figura_comparacao_formula(estacoes_zee, caminho_formula)
    print(f"  {caminho_formula.name}")

    # -----------------------------------------------------------------
    # Gráficos de dispersão propriedade x propriedade: c×T, S×T, σ×T,
    # c×S, σ×S — versão função (k=0) e versão derivada (k=1)
    # -----------------------------------------------------------------
    print("\nGerando gráficos de dispersão propriedade x propriedade...")

    for regiao, zonas in estacoes_processadas.items():
        if not zonas:
            continue
        curvas = {zona: {"dados": df_z, "cor": CORES_ZONA.get(zona, "black")}
                  for zona, df_z in zonas.items()}

        caminho = PASTA_SAIDA / f"dispersao_regiao_{regiao}.png"
        gerar_figura_dispersao(curvas, caminho, titulo=f"Diagramas Propriedade × Propriedade — Região {regiao}")

        caminho_deriv = PASTA_SAIDA / f"dispersao_regiao_{regiao}_derivada.png"
        gerar_figura_dispersao_derivada(
            curvas, caminho_deriv,
            titulo=f"Diagramas Propriedade × Propriedade (Derivadas) — Região {regiao}",
        )
        print(f"  {regiao}: {caminho.name}, {caminho_deriv.name}")

    for zona in ["Rasa", "Plataforma", "ZEE"]:
        curvas = {}
        for regiao, zonas in estacoes_processadas.items():
            if zona in zonas:
                curvas[regiao] = {"dados": zonas[zona], "cor": CORES_REGIAO.get(regiao, "black")}
        if not curvas:
            continue

        caminho = PASTA_SAIDA / f"dispersao_zona_{zona}.png"
        gerar_figura_dispersao(curvas, caminho, titulo=f"Diagramas Propriedade × Propriedade — Zona {zona}")

        caminho_deriv = PASTA_SAIDA / f"dispersao_zona_{zona}_derivada.png"
        gerar_figura_dispersao_derivada(
            curvas, caminho_deriv,
            titulo=f"Diagramas Propriedade × Propriedade (Derivadas) — Zona {zona}",
        )
        print(f"  {zona}: {caminho.name}, {caminho_deriv.name}")

    # -----------------------------------------------------------------
    # Tabela-resumo
    # -----------------------------------------------------------------
    print("\nResumo geral:")
    linhas_resumo = []
    for regiao, zonas in estacoes_processadas.items():
        for zona, estacao in zonas.items():
            def faixa(col):
                if col not in estacao.columns or estacao[col].dropna().empty:
                    return "N/A"
                return (f"{estacao[col].min():.2f} / "
                        f"{estacao[col].mean():.2f} / "
                        f"{estacao[col].max():.2f}")

            linhas_resumo.append({
                "Região": regiao,
                "Zona": zona,
                "N pontos": len(estacao),
                "Prof. máx [m]": estacao["Profundidade [m]"].max(),
                "T mín/méd/máx [°C]": faixa("Temperatura [°c]"),
                "S mín/méd/máx [psu]": faixa("Salinidade [psu]"),
                "c mín/méd/máx [m/s]": faixa("Velocidade do som [m/s]"),
                "ρ mín/méd/máx [kg/m³]": faixa("Densidade ro [kg/m³]"),
                "ρθ mín/méd/máx [kg/m³]": faixa("Densidade Potencial [kg/m³]"),
                "σ mín/méd/máx [S/m]": faixa("Condutividade [S/m]"),
                "P mín/méd/máx [db]": faixa("Pressão [db]"),
            })

    resumo = pd.DataFrame(linhas_resumo)
    print(resumo.to_string(index=False))

    caminho_resumo = PASTA_SAIDA / "resumo_geral.csv"
    resumo.to_csv(caminho_resumo, sep=";", index=False)
    print(f"\nTabela-resumo salva em: {caminho_resumo.resolve()}")


if __name__ == "__main__":
    main()