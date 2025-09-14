# agentes.py

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Literal
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from vrva_funcoes import (
    extrair_zip_para_temp,
    detectar_e_carregar,
    calcular_base_final,
    escrever_planilha_padrao_xlsx_bytes,
)

log = logging.getLogger("vrva.agentes")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

# ─────────────── Entrada/Saída ───────────────
@dataclass
class RodadaEntrada:
    zip_bytes: bytes
    competencia: str
    pct_empresa: float
    usar_llm: bool
    provedor: Literal["gemini", "openai", "openrouter"]
    modelo: Optional[str] = None
    gemini_key: Optional[str] = None
    openai_key: Optional[str] = None
    openrouter_key: Optional[str] = None

@dataclass
class Resultado:
    base_final: "pd.DataFrame"
    validacoes: "pd.DataFrame"
    relatorio: str

@dataclass
class SaidaPipeline:
    tmpdir: str
    resultado: Resultado
    xlsx_bytes: bytes

# ─────────────── LLM factory ───────────────
def _escolher_llm(entrada: RodadaEntrada):
    prov = (entrada.provedor or "gemini").lower()
    modelo = entrada.modelo

    if prov == "gemini":
        nome = modelo or "gemini-1.5-flash"
        if not entrada.gemini_key:
            raise RuntimeError("GEMINI_API_KEY não informada.")
        return ChatGoogleGenerativeAI(model=nome, google_api_key=entrada.gemini_key, temperature=0)
    elif prov == "openai":
        nome = modelo or "gpt-4o-mini"
        if not entrada.openai_key:
            raise RuntimeError("OPENAI_API_KEY não informada.")
        return ChatOpenAI(model=nome, temperature=0, api_key=entrada.openai_key)
    else:  # openrouter
        nome = modelo or "deepseek/deepseek-r1-0528"
        if not entrada.openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY não informada.")
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=entrada.openrouter_key,
            model=nome,
            temperature=0,
            max_tokens=2048,
        )

# ─────────────── nós (agentes) ───────────────
def no_ingestao(ctx: dict) -> dict:
    entrada: RodadaEntrada = ctx["entrada"]
    tmpdir = extrair_zip_para_temp(entrada.zip_bytes)
    dfs = detectar_e_carregar(tmpdir)
    ctx.update({"tmpdir": tmpdir, "dfs": dfs})
    return ctx

def no_validacoes(ctx: dict) -> dict:
    entrada: RodadaEntrada = ctx["entrada"]
    dfs = ctx["dfs"]

    problemas = []
    try:
        assert "matricula" in dfs.ativos.columns
    except Exception:
        problemas.append("Planilha ATIVOS sem coluna 'Matricula' padronizada.")
    if dfs.base_dias_uteis.empty:
        problemas.append("Base de DIAS ÚTEIS vazia ou não reconhecida.")
    if dfs.base_sindicato_valor.empty:
        problemas.append("Base SINDICATO x VALOR vazia ou não reconhecida.")

    if problemas and entrada.usar_llm:
        llm = _escolher_llm(entrada)
        prompt = (
            "Você é um agente de validação de dados. Explique em português e de forma objetiva "
            "os problemas abaixo e como o usuário pode corrigir:\n\n- " + "\n- ".join(problemas)
        )
        explic = llm.invoke([HumanMessage(content=prompt)]).content
    else:
        explic = "OK. Nenhum problema crítico detectado." if not problemas else "\n".join(problemas)

    ctx.update({"problemas": problemas, "explic_validacoes": explic})
    return ctx

def no_calculo(ctx: dict) -> dict:
    entrada: RodadaEntrada = ctx["entrada"]
    dfs = ctx["dfs"]
    base_final, validacoes = calcular_base_final(dfs, entrada.competencia, entrada.pct_empresa)
    ctx.update({"base_final": base_final, "validacoes": validacoes})
    return ctx

def no_exportacao(ctx: dict) -> dict:
    xlsx_bytes = escrever_planilha_padrao_xlsx_bytes(ctx["base_final"], ctx["validacoes"])
    ctx.update({"xlsx_bytes": xlsx_bytes})
    return ctx

def no_explicacao(ctx: dict) -> dict:
    entrada: RodadaEntrada = ctx["entrada"]
    dfs = ctx["dfs"]
    base_final = ctx["base_final"]
    validacoes = ctx["validacoes"]

    rel_deterministico = [
        f"Competência: {entrada.competencia}",
        f"% empresa: {entrada.pct_empresa:.2f}",
        f"Registros finais: {len(base_final)}",
        f"Arquivos: admissoes={len(dfs.admissoes)}, ativos={len(dfs.ativos)}, desligados={len(dfs.desligados)}, ferias={len(dfs.ferias)}",
        f"Base dias úteis: {len(dfs.base_dias_uteis)} | Base sindicato x valor: {len(dfs.base_sindicato_valor)}",
    ]
    texto = "\n".join(rel_deterministico)

    if entrada.usar_llm:
        llm = _escolher_llm(entrada)
        sys = SystemMessage(content="Você é um assistente que gera relatórios claros e executivos.")
        hm = HumanMessage(content=f"Dados de validação:\n{validacoes.to_string(index=False)}\n\nResumo de execução:\n{texto}")
        try:
            texto = llm.invoke([sys, hm]).content
        except Exception as e:
            log.warning(f"LLM falhou ao gerar explicação; usando relatório padrão. Erro: {e}")

    ctx.update({"relatorio": texto})
    return ctx

# ─────────────── grafo ───────────────
def construir_grafo():
    g = StateGraph(dict)
    g.add_node("ingestao", no_ingestao)
    g.add_node("validacoes", no_validacoes)
    g.add_node("calculo", no_calculo)
    g.add_node("exportacao", no_exportacao)
    g.add_node("explicacao", no_explicacao)

    g.set_entry_point("ingestao")
    g.add_edge("ingestao", "validacoes")
    g.add_edge("validacoes", "calculo")
    g.add_edge("calculo", "exportacao")
    g.add_edge("exportacao", "explicacao")
    g.add_edge("explicacao", END)

    app = g.compile()

    class _Wrapper:
        def invoke(self, entrada: RodadaEntrada) -> SaidaPipeline:
            estado = {"entrada": entrada}
            final = app.invoke(estado)
            return SaidaPipeline(
                tmpdir=final["tmpdir"],
                resultado=Resultado(
                    base_final=final["base_final"],
                    validacoes=final["validacoes"],
                    relatorio=final["relatorio"],
                ),
                xlsx_bytes=final["xlsx_bytes"],
            )

    return _Wrapper()
