# vrva_funcoes.py

from __future__ import annotations

import io
import re
import tempfile
import unicodedata
import zipfile
import logging
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("vrva.dominio")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

# ───────────── util ─────────────
def normalizar_txt(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def serie_para_int_seguro(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def serie_para_data_segura(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalizar_txt(c) for c in out.columns]
    return out

def _to_number_br(x) -> Optional[float]:
    """Converte 'R$ 37,50' / '37,50' / '37.50' para float com segurança."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"[^\d,.\-]", "", s)  # remove moeda e espaços
    # caso típico BR: vírgula decimal
    if s.count(",") == 1 and (s.count(".") == 0 or s.find(",") > s.find(".")):
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.to_numeric(s, errors="coerce")

def sindicato_para_estado(s: str) -> str:
    """
    Heurística tolerante para mapear a descrição do sindicato para o estado usado na
    tabela 'Base sindicato x valor'.
    """
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    t = " ".join(t.split())
    if ("SINDPD SP" in t) or (" ESTADO DE SP" in t) or (" SP " in t) or t.endswith(" DE SP.") \
       or ("SIND.TRAB.EM PROC DADOS ESTADO DE SP" in t):
        return "Sao Paulo"
    if ("SINDPPD RS" in t) or (" RIO GRANDE DO SUL" in t) or (" RS " in t):
        return "Rio Grande do Sul"
    if ("SINDPD RJ" in t) or (" RIO DE JANEIRO" in t) or (" RJ " in t):
        return "Rio de Janeiro"
    if (("SIND" in t) and (" PARANA" in t or " PR " in t or " CURITIBA" in t)) or ("SITEPD PR" in t):
        return "Parana"
    # fallback conservador
    return "Sao Paulo"

def df_para_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.fillna("")
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out


# ───────────── leitura ─────────────
def ler_xlsx(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, dtype=object, **kwargs)

def carregar_ativos(path: str | Path) -> pd.DataFrame:
    df = ler_xlsx(path, sheet_name="ATIVOS")
    df = padronizar_colunas(df)
    keep = [c for c in df.columns if c in {"matricula", "titulo_do_cargo", "desc_situacao", "sindicato"}]
    df = df[keep].copy()
    df["matricula"] = serie_para_int_seguro(df["matricula"])
    return df.dropna(subset=["matricula"]).reset_index(drop=True)

def carregar_desligados(path: str | Path) -> pd.DataFrame:
    df = ler_xlsx(path)
    df = padronizar_colunas(df)
    mcol = next((c for c in df.columns if "matricula" in c), None)
    dcol = next((c for c in df.columns if "demiss" in c or "deslig" in c or "data" in c), None)
    ccol = next((c for c in df.columns if "comunicado" in c), None)
    df = df[[mcol, dcol, ccol]].copy()
    df.columns = ["matricula", "data_desligamento", "comunicado"]
    df["matricula"] = serie_para_int_seguro(df["matricula"])
    df["data_desligamento"] = serie_para_data_segura(df["data_desligamento"])
    df["comunic_ok"] = df["comunicado"].astype(str).str.contains("ok", case=False, na=False)
    return df.dropna(subset=["matricula"]).reset_index(drop=True)

def carregar_ferias(path: str | Path) -> pd.DataFrame:
    df = ler_xlsx(path)
    df = padronizar_colunas(df)
    mcol = next((c for c in df.columns if "matricula" in c), None)
    dcol = next((c for c in df.columns if "dia" in c), None)
    df = df[[mcol, dcol]].copy()
    df.columns = ["matricula", "dias_ferias"]
    df["matricula"] = serie_para_int_seguro(df["matricula"])
    df["dias_ferias"] = pd.to_numeric(df["dias_ferias"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["matricula"]).reset_index(drop=True)

def carregar_exterior(path: str | Path) -> pd.DataFrame:
    """
    Lê EXTERIOR.xlsx (colunas: Cadastro/Matricula, Valor, Comentário/Obs).
    Registros com 'remov' no comentário são descartados; os demais são usados
    como override de TOTAL (valor absoluto informado) no cálculo.
    """
    df = ler_xlsx(path)
    df = padronizar_colunas(df)
    mcol = next((c for c in df.columns if "matricula" in c or "cadastro" in c), None)
    vcol = next((c for c in df.columns if "valor" in c), None)
    ocol = next((c for c in df.columns if "obs" in c or "coment" in c or "status" in c), None)
    if mcol is None:
        return pd.DataFrame(columns=["matricula", "valor", "obs"])
    out = pd.DataFrame({
        "matricula": serie_para_int_seguro(df[mcol]),
        "valor": df[vcol] if vcol else 0,
        "obs": (df[ocol].astype(str) if ocol else "")
    })
    # normaliza valores monetários
    out["valor"] = out["valor"].apply(_to_number_br).fillna(0.0).astype(float)
    # descarta linhas explicitamente marcadas como removidas
    mask_keep = ~out["obs"].astype(str).str.lower().str.contains("remov")
    out = out[mask_keep]
    return out.dropna(subset=["matricula"]).reset_index(drop=True)

def carregar_lista_matriculas(path: str | Path) -> pd.DataFrame:
    df = ler_xlsx(path)
    df = padronizar_colunas(df)
    mcol = next((c for c in df.columns if "matricula" in c or "cadastro" in c), None)
    if mcol is None:
        return pd.DataFrame(columns=["matricula"])
    out = pd.DataFrame({"matricula": serie_para_int_seguro(df[mcol])}).dropna(subset=["matricula"])
    return out.reset_index(drop=True)

def carregar_admissoes(path: str | Path) -> pd.DataFrame:
    df = ler_xlsx(path)
    df = padronizar_colunas(df)
    mcol = next((c for c in df.columns if "matricula" in c), "matricula")
    acol = next((c for c in df.columns if "admiss" in c), None)
    cargo_col = next((c for c in df.columns if "cargo" in c or "funcao" in c or "titulo" in c), None)

    keep = [mcol]
    if acol: keep.append(acol)
    if cargo_col: keep.append(cargo_col)
    df = df[keep].copy()

    df.columns = ["matricula"] + (["data_admissao"] if acol else []) + (["cargo"] if cargo_col else [])
    df["matricula"] = serie_para_int_seguro(df["matricula"])
    if acol:
        df["data_admissao"] = serie_para_data_segura(df["data_admissao"])

    # limpa anotações indicativas de inelegibilidade
    if "cargo" in df.columns:
        cargo_txt = df["cargo"].astype(str).str.lower()
        mask_bad = cargo_txt.str.contains("demit", na=False) | cargo_txt.str.contains("nao recebe vr", na=False)
        df = df[~mask_bad].copy()

    return df.dropna(subset=["matricula"]).reset_index(drop=True)

def carregar_base_dias_uteis(path: str | Path) -> pd.DataFrame:
    raw = ler_xlsx(path, header=None)
    header_row = None
    for i in range(min(6, len(raw))):
        joined = " ".join(str(x) for x in raw.iloc[i, :2].tolist())
        if "SIND" in joined.upper() and "DIA" in joined.upper():
            header_row = i
            break
    if header_row is None:
        header_row = 1 if len(raw) > 1 else 0

    df = ler_xlsx(path, header=header_row)
    cols = [c for c in df.columns if not str(c).startswith("Unnamed")]
    if len(cols) >= 2:
        df = df[cols[:2]].copy()
        df.columns = ["sindicato", "dias_uteis"]
    else:
        df = raw.iloc[header_row + 1 :, :2].copy()
        df.columns = ["sindicato", "dias_uteis"]

    df["sindicato"] = df["sindicato"].astype(str).str.strip()
    df = df[df["sindicato"].str.len() > 0]
    df["dias_uteis"] = pd.to_numeric(df["dias_uteis"], errors="coerce").astype(float).round().astype(int)
    return df.reset_index(drop=True)

def carregar_base_sindicato_valor(path: str | Path) -> pd.DataFrame:
    df = ler_xlsx(path, header=0)
    non_empty = [c for c in df.columns if not str(c).startswith("Unnamed")]
    if len(non_empty) < 2:
        non_empty = list(df.columns)[:2]
    df = df[non_empty[:2]].copy()
    df.columns = ["estado", "valor"]
    df["estado"] = df["estado"].astype(str).str.strip()
    df = df[df["estado"].str.len() > 0]

    # normaliza moeda/decimal
    df["valor"] = df["valor"].apply(_to_number_br)
    df = df.dropna(subset=["valor"])
    df["valor"] = df["valor"].astype(float)

    return df.reset_index(drop=True)

def carregar_afastamentos(path: str | Path) -> pd.DataFrame:
    """
    Lê AFASTAMENTOS.xlsx e tenta extrair 'retorno em dd/mm' da coluna de observações.
    Retorna: matricula (Int64), retorno_dia (Int64), retorno_mes (Int64), obs (str).
    """
    df = ler_xlsx(path)
    df = padronizar_colunas(df)
    mcol = next((c for c in df.columns if "matricula" in c or "cadastro" in c), None)
    ocol = next((c for c in df.columns if "na_compra" in c or "obs" in c or "coment" in c or "status" in c), None)
    if mcol is None:
        return pd.DataFrame(columns=["matricula", "retorno_dia", "retorno_mes", "obs"])

    out = pd.DataFrame({"matricula": serie_para_int_seguro(df[mcol])})
    obs = df[ocol].astype(str) if ocol else ""
    out["obs"] = obs

    # extrai 'dd/mm' (aceita '-' também)
    m = obs.str.extract(r"(\d{1,2})[/-](\d{1,2})", expand=True)
    out["retorno_dia"] = pd.to_numeric(m[0], errors="coerce").astype("Int64")
    out["retorno_mes"] = pd.to_numeric(m[1], errors="coerce").astype("Int64")

    return out.dropna(subset=["matricula"]).reset_index(drop=True)


# ───────────── detecção & carga ─────────────
@dataclass
class Entradas:
    admissoes: pd.DataFrame
    ativos: pd.DataFrame
    desligados: pd.DataFrame
    ferias: pd.DataFrame
    afastamentos: pd.DataFrame
    exterior: pd.DataFrame        # contém colunas: matricula, valor, obs
    aprendiz: pd.DataFrame
    estagio: pd.DataFrame
    base_dias_uteis: pd.DataFrame
    base_sindicato_valor: pd.DataFrame
    vr_modelo: Optional[pd.DataFrame]

def extrair_zip_para_temp(buff: bytes) -> str:
    tmp = tempfile.mkdtemp(prefix="vrva_")
    with zipfile.ZipFile(io.BytesIO(buff)) as zf:
        zf.extractall(tmp)
    log.info("ZIP extraído em %s", tmp)
    return tmp

def detectar_e_carregar(pasta: str) -> Entradas:
    arquivos: Dict[str, Path] = {}
    for p in Path(pasta).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".xlsx", ".xls"}:
            arquivos[p.name] = p

    def _achar(*chaves: str) -> Optional[Path]:
        want = [normalizar_txt(k) for k in chaves]
        for nome, path in arquivos.items():
            nk = normalizar_txt(nome)
            if all(w in nk for w in want):
                return path
        return None

    def _obrigatorio(keys: Tuple[str, ...], nice: str) -> Path:
        p = _achar(*keys)
        if not p:
            disponiveis = ", ".join(sorted(arquivos))
            raise RuntimeError(f"Arquivo obrigatório não encontrado: {nice} (achados: {disponiveis})")
        return p

    p_adm = _obrigatorio(("admiss",), "ADMISSÃO ABRIL.xlsx")
    p_ativos = _obrigatorio(("ativos",), "ATIVOS.xlsx")
    p_des = _obrigatorio(("deslig",), "DESLIGADOS.xlsx")
    p_fer = _obrigatorio(("ferias",), "FÉRIAS.xlsx")
    p_du = _obrigatorio(("base", "dias", "uteis"), "Base dias uteis.xlsx")
    p_bsv = _obrigatorio(("base", "sindicato", "valor"), "Base sindicato x valor.xlsx")

    p_afr = _achar("afast")
    p_ext = _achar("exterior")
    p_apr = _achar("aprend")
    p_est = _achar("estag")
    p_vr = _achar("vr", "mensal")

    admissoes = carregar_admissoes(p_adm);       log.info("admissoes: %d linhas", len(admissoes))
    ativos = carregar_ativos(p_ativos);          log.info("ativos: %d linhas", len(ativos))
    desligados = carregar_desligados(p_des);     log.info("desligados: %d linhas", len(desligados))
    ferias = carregar_ferias(p_fer);             log.info("ferias: %d linhas", len(ferias))
    afast = carregar_afastamentos(p_afr) if p_afr else pd.DataFrame(columns=["matricula","retorno_dia","retorno_mes","obs"]);   log.info("afastamentos: %d linhas", len(afast))
    exterior = carregar_exterior(p_ext) if p_ext else pd.DataFrame(columns=["matricula", "valor", "obs"]); log.info("exterior: %d linhas", len(exterior))
    aprendiz = carregar_lista_matriculas(p_apr) if p_apr else pd.DataFrame(columns=["matricula"]); log.info("aprendiz: %d linhas", len(aprendiz))
    estagio = carregar_lista_matriculas(p_est) if p_est else pd.DataFrame(columns=["matricula"]);  log.info("estagio: %d linhas", len(estagio))

    try:
        base_dias_uteis = carregar_base_dias_uteis(p_du)
        log.info("base_dias_uteis: %d linhas", len(base_dias_uteis))
    except Exception as e:
        log.exception("Falha ao ler BASE DIAS ÚTEIS: %s", e)
        raise RuntimeError(f"Falha ao ler BASE DIAS ÚTEIS: {e}")

    try:
        base_sind_valor = carregar_base_sindicato_valor(p_bsv)
        log.info("base_sindicato_valor: %d linhas", len(base_sind_valor))
    except Exception as e:
        log.exception("Falha ao ler BASE SINDICATO x VALOR: %s", e)
        raise RuntimeError(f"Falha ao ler BASE SINDICATO x VALOR: {e}")

    vr_modelo = None
    if p_vr:
        try:
            vr_modelo = ler_xlsx(p_vr)
            log.info("vr_modelo: %d linhas", len(vr_modelo))
        except Exception:
            log.warning("Não foi possível ler a planilha modelo VR; seguiremos sem ela.")

    return Entradas(
        admissoes=admissoes,
        ativos=ativos,
        desligados=desligados,
        ferias=ferias,
        afastamentos=afast,
        exterior=exterior,
        aprendiz=aprendiz,
        estagio=estagio,
        base_dias_uteis=base_dias_uteis,
        base_sindicato_valor=base_sind_valor,
        vr_modelo=vr_modelo,
    )


# ───────────── cálculo ─────────────
def parse_competencia(competencia: str) -> Tuple[int, int, str]:
    """Retorna (ano, mes, 'MM/YYYY'). Aceita 'YYYY-MM' ou 'MM/YYYY'."""
    if "/" in competencia and len(competencia) == 7:
        m, y = competencia.split("/")
        y, m = int(y), int(m)
    else:
        y, m = map(int, competencia.split("-"))
    return y, m, f"{m:02d}/{y}"

def calcular_base_final(dfs: Entradas, competencia: str, pct_empresa: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ano, mes, competencia_str = parse_competencia(competencia)
    days_in_month = monthrange(ano, mes)[1]
    first_day = date(ano, mes, 1)
    last_day = date(ano, mes, days_in_month)

    base = dfs.ativos.copy()
    # só quem está Trabalhando
    base = base[base["desc_situacao"].astype(str).str.contains("trabal", case=False, na=False)].copy()
    base["cargo"] = base["titulo_do_cargo"].astype(str)

    # exclusões duras: aprendiz/estágio
    excl = set(dfs.aprendiz["matricula"].dropna().astype(int).tolist()) \
        | set(dfs.estagio["matricula"].dropna().astype(int).tolist())
    base = base[~base["matricula"].astype(int).isin(excl)].copy()
    base = base[~base["cargo"].str.contains("diretor", case=False, na=False)].copy()

    # férias (sem período)
    base = base.merge(dfs.ferias[["matricula", "dias_ferias"]], on="matricula", how="left")
    base["dias_ferias"] = base["dias_ferias"].fillna(0).astype(int)

    # desligados + comunicado
    base = base.merge(dfs.desligados[["matricula", "data_desligamento", "comunic_ok"]], on="matricula", how="left")

    # afastamentos (com possível retorno dentro da competência)
    if not dfs.afastamentos.empty:
        cols = [c for c in ["matricula","retorno_dia","retorno_mes","obs"] if c in dfs.afastamentos.columns]
        af = dfs.afastamentos[cols].copy()
        base = base.merge(af, on="matricula", how="left")

    # dias úteis por sindicato (normalização robusta)
    du = dfs.base_dias_uteis.copy()
    base["sind_norm"] = base["sindicato"].astype(str).apply(normalizar_txt)
    du["sind_norm"] = du["sindicato"].astype(str).apply(normalizar_txt)
    base = base.merge(du[["sind_norm", "dias_uteis"]], on="sind_norm", how="left")
    base["dias_uteis"] = base["dias_uteis"].fillna(22).astype(int)

    # valor do dia (por estado heurístico)
    val_map = {normalizar_txt(row["estado"]): row["valor"] for _, row in dfs.base_sindicato_valor.iterrows()}

    def valor_dia(sind: str) -> float:
        est = normalizar_txt(sindicato_para_estado(sind))
        return float(val_map.get(est, 35.0))

    base["valor_dia"] = base["sindicato"].astype(str).apply(valor_dia).astype(float)

    # admissões
    adm = dfs.admissoes[["matricula", "data_admissao"]].copy()
    base = base.merge(adm, on="matricula", how="left")

    # regra de dias a pagar via janela trabalhada (start–end) e proporcional sobre DIAS ÚTEIS
    def calcular_dias(row) -> int:
        # afastado e com retorno fora do mês → 0
        if "retorno_mes" in row and pd.notna(row["retorno_mes"]) and int(row["retorno_mes"]) != mes:
            return 0

        start = first_day
        end = last_day

        # admissão no mês
        da = row.get("data_admissao", None)
        if pd.notna(da) and isinstance(da, date) and da.year == ano and da.month == mes:
            start = max(start, da)

        # retorno de afastamento dentro do mês
        if "retorno_dia" in row and pd.notna(row["retorno_dia"]) and (pd.isna(row.get("retorno_mes")) or int(row["retorno_mes"]) == mes):
            try:
                ret = date(ano, mes, int(row["retorno_dia"]))
                start = max(start, ret)
            except Exception:
                pass

        # desligamento no mês (regras do enunciado)
        dd = row.get("data_desligamento", None)
        if pd.notna(dd) and isinstance(dd, date) and dd.year == ano and dd.month == mes:
            if bool(row.get("comunic_ok", False)) and dd.day <= 15:
                return 0
            if (not bool(row.get("comunic_ok", False))) and dd.day <= 15:
                end = last_day  # integral
            else:
                end = min(dd, last_day)

        if end < start:
            return 0

        dias_calend = (end - start).days + 1
        frac = dias_calend / days_in_month
        dias_prop = int(round(int(row["dias_uteis"]) * frac))
        dias_final = dias_prop - int(row.get("dias_ferias", 0))
        return max(0, dias_final)

    base["dias"] = base.apply(calcular_dias, axis=1)

    # totais
    base["total"] = (base["dias"].astype(float) * base["valor_dia"].astype(float)).round(2)

    # Overrides de EXTERIOR (valor absoluto fornecido na planilha)
    if not dfs.exterior.empty:
        ext = dfs.exterior[["matricula", "valor", "obs"]].copy()
        ext["matricula"] = ext["matricula"].astype("Int64")
        base = base.merge(ext, on="matricula", how="left", suffixes=("", "_ext"))
        mask_ext = pd.to_numeric(base["valor"], errors="coerce").fillna(0.0) > 0
        base.loc[mask_ext, "total"] = base.loc[mask_ext, "valor"].astype(float).round(2)
        base.drop(columns=["valor"], inplace=True, errors="ignore")

    # rateio empresa/profissional
    base["custo_empresa"] = (base["total"] * float(pct_empresa)).round(2)
    base["desconto_profissional"] = (base["total"] - base["custo_empresa"]).round(2)

    # validações (mantém EXACTAMENTE os nomes do modelo do professor)
    def contar(cond) -> int:
        return int(cond.sum()) if hasattr(cond, "sum") else int(cond)

    dd_series = pd.to_datetime(base["data_desligamento"], errors="coerce")
    dd_mes_comp = dd_series.dt.month.eq(mes) & dd_series.notna()
    deslig_dia_le15 = (dd_mes_comp & dd_series.dt.day.le(15)).fillna(False)
    deslig_dia_ge16 = (dd_mes_comp & dd_series.dt.day.ge(16)).fillna(False)

    valid = pd.DataFrame([
        {"Validações": "Afastados / Licenças", "Check": len(dfs.afastamentos)},
        {"Validações": "DESLIGADOS GERAL", "Check": len(dfs.desligados)},
        {"Validações": "Admitidos mês", "Check": contar(pd.to_datetime(dfs.admissoes["data_admissao"], errors="coerce").dt.month.eq(mes))},
        {"Validações": "Férias", "Check": len(dfs.ferias)},
        {"Validações": "ESTAGIARIO", "Check": len(dfs.estagio)},
        {"Validações": "APRENDIZ", "Check": len(dfs.aprendiz)},
        {"Validações": "SINDICATOS x VALOR", "Check": len(dfs.base_sindicato_valor)},
        {"Validações": "DESLIGADOS ATÉ O DIA 15 DO MÊS - SE JÁ ESTIVEREM CIENTES DO DESLIGAMENTO EXCLUIR DA COMPRA - SE NÃO TIVER O OK COMPRAR INTEGRAL",
         "Check": contar(deslig_dia_le15)},
        {"Validações": "DESLIGADOS DO DIA 16 ATÉ O ULTIMO DIA DO MÊS PODE FAZER A RECARGA CHEIA E DEIXAR O DESCONTO PROPORCIONAL PARA SER FEITO EM RESCISÃO",
         "Check": contar(deslig_dia_ge16)},
        {"Validações": "ATENDIMENTOS/OBS", "Check": 0},
        {"Validações": "Admitidos mês anterior", "Check": contar(pd.to_datetime(dfs.admissoes["data_admissao"], errors="coerce").dt.month.eq((mes - 1) if mes > 1 else 12))},
        {"Validações": "EXTERIOR", "Check": len(dfs.exterior)},
        {"Validações": "ATIVOS", "Check": len(dfs.ativos)},
        {"Validações": "REVISAR O CALCULO DE PGTO SE ESTÁ CORRETO ANTES DE GERAR OS VALES", "Check": len(base)},
    ])

    # layout final (10 colunas exatamente)
    base_final = pd.DataFrame({
        "Matricula": base["matricula"].astype("Int64"),
        "Admissao": base.get("data_admissao"),
        "Sindicato do Colaborador": base["sindicato"],
        "Competencia": competencia_str,  # MM/YYYY
        "Dias": base["dias"].astype(int),
        "VALOR DIARIO VR": base["valor_dia"].astype(float),
        "TOTAL": base["total"].astype(float),
        "Custo empresa": base["custo_empresa"].astype(float),
        "Desconto profissional": base["desconto_profissional"].astype(float),
        "OBS GERAL": pd.Series([""] * len(base)),
    })

    return base_final, valid


# ───────────── saneamento & writer XLSX ─────────────
def sanear_saida_planilha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que:
      - 10 colunas com nomes e ordem corretos
      - números verdadeiramente numéricos (evita Excel tratar como texto)
      - dias >= 0, valores não-negativos e arredondados a 2 casas
      - sem 'null' string em datas/obs
    """
    colunas = [
        "Matricula", "Admissao", "Sindicato do Colaborador", "Competencia",
        "Dias", "VALOR DIARIO VR", "TOTAL", "Custo empresa",
        "Desconto profissional", "OBS GERAL"
    ]
    df = df.copy()
    df = df[colunas].copy()

    df["Matricula"] = pd.to_numeric(df["Matricula"], errors="coerce").astype("Int64")
    df["Dias"] = pd.to_numeric(df["Dias"], errors="coerce").fillna(0).astype(int)
    df.loc[df["Dias"] < 0, "Dias"] = 0

    for c in ["VALOR DIARIO VR", "TOTAL", "Custo empresa", "Desconto profissional"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float).round(2)
        df.loc[df[c] < 0, c] = 0.0

    # Data e texto limpos
    df["Admissao"] = pd.to_datetime(df["Admissao"], errors="coerce").dt.date
    for c in ["Sindicato do Colaborador", "Competencia", "OBS GERAL"]:
        df[c] = df[c].astype(str).replace({"None": "", "nan": "", "NaT": "", "null": ""}, regex=True)

    return df


def escrever_planilha_padrao_xlsx_bytes(base_final: pd.DataFrame, validacoes: pd.DataFrame) -> bytes:
    """
    XLSX no padrão:
      • 'VR MENSAL': header A1:J1 com fundo preto, dados a partir da linha 2,
        formatos numéricos, filtros no cabeçalho.
      • 'Validações': cabeçalho A1:B1 com fundo preto.
    """
    dados = sanear_saida_planilha(base_final)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        cols_ordem = [
            "Matricula", "Admissao", "Sindicato do Colaborador", "Competencia",
            "Dias", "VALOR DIARIO VR", "TOTAL", "Custo empresa",
            "Desconto profissional", "OBS GERAL"
        ]
        dados[cols_ordem].to_excel(xw, sheet_name="VR MENSAL", index=False, header=False, startrow=1, startcol=0)
        wb = xw.book
        ws = xw.sheets["VR MENSAL"]

        fmt_header = wb.add_format({
            "bold": True, "font_color": "white", "bg_color": "black",
            "align": "center", "valign": "vcenter", "border": 1
        })
        fmt_int = wb.add_format({"num_format": "0"})
        fmt_money = wb.add_format({"num_format": "R$ #,##0.00"})
        fmt_date = wb.add_format({"num_format": "dd/mm/yyyy"})
        fmt_text_center = wb.add_format({"align": "center"})

        # Cabeçalho manual
        ws.write_row(0, 0, cols_ordem, fmt_header)

        # Larguras e formatos
        widths = {"A": 10, "B": 12, "C": 95, "D": 12, "E": 6, "F": 14, "G": 14, "H": 14, "I": 18, "J": 40}
        for col, w in widths.items():
            ws.set_column(f"{col}:{col}", w)

        nrows = len(dados)
        ws.set_column("A:A", widths["A"], fmt_int)          # Matricula
        ws.set_column("B:B", widths["B"], fmt_date)         # Admissao
        ws.set_column("C:C", widths["C"], None)             # Sindicato
        ws.set_column("D:D", widths["D"], fmt_text_center)  # Competencia (texto MM/YYYY)
        ws.set_column("E:E", widths["E"], fmt_int)          # Dias
        ws.set_column("F:F", widths["F"], fmt_money)        # VALOR DIARIO VR
        ws.set_column("G:G", widths["G"], fmt_money)        # TOTAL
        ws.set_column("H:H", widths["H"], fmt_money)        # Custo empresa
        ws.set_column("I:I", widths["I"], fmt_money)        # Desconto profissional
        ws.set_column("J:J", widths["J"], None)             # OBS

        ws.autofilter(0, 0, nrows, len(cols_ordem) - 1)

        # Validações — nomes idênticos ao modelo do professor
        vdf = validacoes[["Validações", "Check"]].copy()
        vdf.to_excel(xw, sheet_name="Validações", index=False, header=False, startrow=1, startcol=0)
        wsv = xw.sheets["Validações"]
        wsv.write_row(0, 0, ["Validações", "Check"], fmt_header)
        wsv.set_column("A:A", 95)
        wsv.set_column("B:B", 12, fmt_int)
        wsv.autofilter(0, 0, len(vdf), 1)

    out.seek(0)
    return out.getvalue()
