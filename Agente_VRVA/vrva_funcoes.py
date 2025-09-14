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

def sindicato_para_estado(s: str) -> str:
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    if " ESTADO DE SP" in t or " SP " in t or t.startswith("SINDPD SP") or " DE SP." in t:
        return "Sao Paulo"
    if " RIO GRANDE DO SUL" in t or " RS " in t or "SINDPPD RS" in t:
        return "Rio Grande do Sul"
    if " RIO DE JANEIRO" in t or " RJ " in t:
        return "Rio de Janeiro"
    if " CURITIBA" in t or " PARANA" in t or " PR " in t:
        return "Parana"
    return "Sao Paulo"

def df_para_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
    mcol = "matricula"
    acol = next((c for c in df.columns if "admiss" in c), None)
    df = df[[mcol, acol]].copy()
    df.columns = ["matricula", "data_admissao"]
    df["matricula"] = serie_para_int_seguro(df["matricula"])
    df["data_admissao"] = serie_para_data_segura(df["data_admissao"])
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
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["valor"])
    return df.reset_index(drop=True)

# ───────────── detecção & carga ─────────────
@dataclass
class Entradas:
    admissoes: pd.DataFrame
    ativos: pd.DataFrame
    desligados: pd.DataFrame
    ferias: pd.DataFrame
    afastamentos: pd.DataFrame
    exterior: pd.DataFrame
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
    afast = carregar_lista_matriculas(p_afr) if p_afr else pd.DataFrame(columns=["matricula"]);   log.info("afastamentos: %d linhas", len(afast))
    exterior = carregar_lista_matriculas(p_ext) if p_ext else pd.DataFrame(columns=["matricula"]); log.info("exterior: %d linhas", len(exterior))
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
    if "/" in competencia and len(competencia) == 7:
        m, y = competencia.split("/")
        y, m = int(y), int(m)
    else:
        y, m = map(int, competencia.split("-"))
    return y, m, f"{m:02d}/{y}"

def calcular_base_final(dfs: Entradas, competencia: str, pct_empresa: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ano, mes, competencia_str = parse_competencia(competencia)
    days_in_month = monthrange(ano, mes)[1]

    base = dfs.ativos.copy()
    base = base[base["desc_situacao"].astype(str).str.contains("trabal", case=False, na=False)].copy()
    base["cargo"] = base["titulo_do_cargo"].astype(str)

    excl = set(dfs.aprendiz["matricula"].dropna().astype(int).tolist()) \
        | set(dfs.estagio["matricula"].dropna().astype(int).tolist()) \
        | set(dfs.exterior["matricula"].dropna().astype(int).tolist()) \
        | set(dfs.afastamentos["matricula"].dropna().astype(int).tolist())
    base = base[~base["matricula"].astype(int).isin(excl)].copy()
    base = base[~base["cargo"].str.contains("diretor", case=False, na=False)].copy()

    base = base.merge(dfs.ferias[["matricula", "dias_ferias"]], on="matricula", how="left")
    base["dias_ferias"] = base["dias_ferias"].fillna(0).astype(int)

    base = base.merge(dfs.desligados[["matricula", "data_desligamento", "comunic_ok"]], on="matricula", how="left")

    du = dfs.base_dias_uteis.copy()
    base["sind_norm"] = base["sindicato"].astype(str).apply(normalizar_txt)
    du["sind_norm"] = du["sindicato"].astype(str).apply(normalizar_txt)
    base = base.merge(du[["sind_norm", "dias_uteis"]], on="sind_norm", how="left")
    base["dias_uteis"] = base["dias_uteis"].fillna(22).astype(int)

    val_map = {row["estado"]: row["valor"] for _, row in dfs.base_sindicato_valor.iterrows()}

    def valor_dia(sind: str) -> float:
        est = sindicato_para_estado(sind)
        for k, v in val_map.items():
            if normalizar_txt(k) == normalizar_txt(est):
                return float(v)
        return np.nan

    base["valor_dia"] = base["sindicato"].astype(str).apply(valor_dia).fillna(35.0).astype(float)

    adm = dfs.admissoes[["matricula", "data_admissao"]].copy()
    base = base.merge(adm, on="matricula", how="left")

    def calcular_dias(row) -> int:
        dias = max(0, int(row["dias_uteis"]) - int(row["dias_ferias"]))
        da = row.get("data_admissao", None)
        if pd.notna(da) and isinstance(da, date) and da.year == ano and da.month == mes:
            frac = (days_in_month - da.day + 1) / 30.0
            dias = int(round(dias * frac))
        dd = row.get("data_desligamento", None)
        if pd.notna(dd) and isinstance(dd, date) and dd.year == ano and dd.month == mes:
            if bool(row.get("comunic_ok", False)) and dd.day <= 15:
                return 0
            frac = min(dd.day, days_in_month) / 30.0
            dias = int(round(dias * frac))
        return max(0, dias)

    base["dias"] = base.apply(calcular_dias, axis=1)
    base["total"] = base["dias"] * base["valor_dia"]
    base["custo_empresa"] = base["total"] * float(pct_empresa)
    base["desconto_profissional"] = base["total"] - base["custo_empresa"]

    def contar(cond) -> int:
        return int(cond.sum()) if hasattr(cond, "sum") else int(cond)

    valid = pd.DataFrame([
        {"Validações": "Afastados / Licenças", "Check": contar(dfs.afastamentos["matricula"].notna())},
        {"Validações": "DESLIGADOS GERAL", "Check": len(dfs.desligados)},
        {"Validações": "Admitidos mês", "Check": contar(pd.to_datetime(dfs.admissoes["data_admissao"], errors="coerce").dt.month.eq(mes))},
        {"Validações": "Férias", "Check": len(dfs.ferias)},
        {"Validações": "ESTAGIARIO", "Check": len(dfs.estagio)},
        {"Validações": "APRENDIZ", "Check": len(dfs.aprendiz)},
        {"Validações": "SINDICATOS x VALOR", "Check": len(dfs.base_sindicato_valor)},
        {"Validações": "DESLIGADOS ATÉ O DIA 15 DO MÊS - SE JÁ ESTIVEREM CIENTES DO DESLIGAMENTO EXCLUIR DA COMPRA - SE NÃO TIVER O OK COMPRAR INTEGRAL",
         "Check": contar((pd.to_datetime(base["data_desligamento"], errors="coerce").dt.day.le(15)).fillna(False))},
        {"Validações": "DESLIGADOS DO DIA 16 ATÉ O ULTIMO DIA DO MÊS PODE FAZER A RECARGA CHEIA E DEIXAR O DESCONTO PROPORCIONAL PARA SER FEITO EM RESCISÃO",
         "Check": contar((pd.to_datetime(base["data_desligamento"], errors="coerce").dt.day.ge(16)).fillna(False))},
        {"Validações": "ATENDIMENTOS/OBS", "Check": 0},
        {"Validações": "Admitidos mês anterior", "Check": contar(pd.to_datetime(dfs.admissoes["data_admissao"], errors="coerce").dt.month.eq((mes - 1) if mes > 1 else 12))},
        {"Validações": "EXTERIOR", "Check": len(dfs.exterior)},
        {"Validações": "ATIVOS", "Check": len(dfs.ativos)},
        {"Validações": "REVISAR O CALCULO DE PGTO SE ESTÁ CORRETO ANTES DE GERAR OS VALES", "Check": len(base)},
    ])

    base_final = pd.DataFrame({
        "Matricula": base["matricula"].astype("Int64"),
        "Admissao": base["data_admissao"],
        "Sindicato do Colaborador": base["sindicato"],
        "Competencia": competencia_str,
        "Dias": base["dias"].astype(int),
        "VALOR DIARIO VR": base["valor_dia"].astype(float),
        "TOTAL": base["total"].astype(float),
        "Custo empresa": base["custo_empresa"].astype(float),
        "Desconto profissional": base["desconto_profissional"].astype(float),
        "OBS GERAL": pd.Series([""] * len(base)),
    })

    return base_final, valid

# ───────────── XLSX writer ─────────────
def escrever_planilha_padrao_xlsx_bytes(base_final: pd.DataFrame, validacoes: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        cols_ordem = [
            "Matricula", "Admissao", "Sindicato do Colaborador", "Competencia",
            "Dias", "VALOR DIARIO VR", "TOTAL", "Custo empresa",
            "Desconto profissional", "OBS GERAL"
        ]
        dados = base_final[cols_ordem].copy()
        dados.to_excel(xw, sheet_name="VR MENSAL", index=False, header=False, startrow=1, startcol=0)

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

        ws.write_row(0, 0, cols_ordem, fmt_header)

        widths = {"A": 10, "B": 12, "C": 95, "D": 12, "E": 6, "F": 14, "G": 14, "H": 14, "I": 18, "J": 40}
        for col, w in widths.items():
            ws.set_column(f"{col}:{col}", w)

        nrows = len(dados)
        ws.set_column("A:A", widths["A"], fmt_int)
        ws.set_column("B:B", widths["B"], fmt_date)
        ws.set_column("C:C", widths["C"], None)
        ws.set_column("D:D", widths["D"], fmt_text_center)
        ws.set_column("E:E", widths["E"], fmt_int)
        ws.set_column("F:F", widths["F"], fmt_money)
        ws.set_column("G:G", widths["G"], fmt_money)
        ws.set_column("H:H", widths["H"], fmt_money)
        ws.set_column("I:I", widths["I"], fmt_money)
        ws.set_column("J:J", widths["J"], None)

        ws.autofilter(0, 0, nrows, len(cols_ordem) - 1)

        vdf = validacoes[["Validações", "Check"]].copy()
        vdf.to_excel(xw, sheet_name="Validações", index=False, header=False, startrow=1, startcol=0)
        wsv = xw.sheets["Validações"]
        wsv.write_row(0, 0, ["Validações", "Check"], fmt_header)
        wsv.set_column("A:A", 95)
        wsv.set_column("B:B", 12, fmt_int)
        wsv.autofilter(0, 0, len(vdf), 1)

    out.seek(0)
    return out.getvalue()
