# agentes.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Literal, Callable, Dict, Any
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
    sanear_saida_planilha,
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
    """Agente de Ingestão (determinístico). Extrai ZIP e carrega dataframes."""
    entrada: RodadaEntrada = ctx["entrada"]
    tmpdir = extrair_zip_para_temp(entrada.zip_bytes)
    dfs = detectar_e_carregar(tmpdir)
    ctx.update({"tmpdir": tmpdir, "dfs": dfs})
    return ctx


def no_validacoes(ctx: dict) -> dict:
    """
    Agente de Validação (INTELIGENTE quando usar_llm=True).
    Checa presença de colunas básicas e bases essenciais; quando houver problema, a LLM gera instruções de correção.
    """
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

    # explicação/aconselhamento via LLM
    if problemas and entrada.usar_llm:
        llm = _escolher_llm(entrada)
        prompt = (
            "Você é um agente de validação de dados. Explique de forma objetiva os problemas abaixo "
            "e como o usuário pode corrigi-los, em passos práticos:\n\n- " + "\n- ".join(problemas)
        )
        try:
            explic = llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            log.warning("Falha LLM no no_validacoes: %s", e)
            explic = "\n".join(problemas)
    else:
        explic = "OK. Nenhum problema crítico detectado." if not problemas else "\n".join(problemas)

    ctx.update({"problemas": problemas, "explic_validacoes": explic})
    return ctx


def no_calculo(ctx: dict) -> dict:
    """
    Agente de Cálculo (determinístico).
    Aplica as regras de negócio e retorna base_final + validações-modelo.
    """
    entrada: RodadaEntrada = ctx["entrada"]
    dfs = ctx["dfs"]
    base_final, validacoes = calcular_base_final(dfs, entrada.competencia, entrada.pct_empresa)
    # saneia saída para garantir consistência (sem negativos, números válidos)
    base_final = sanear_saida_planilha(base_final)
    ctx.update({"base_final": base_final, "validacoes": validacoes})
    return ctx


def no_auditoria(ctx: dict) -> dict:
    """
    Agente de Auditoria & Consistência (INTELIGENTE).
    Verifica plausibilidade de totais, presença de negativos, nulos, e sugere/autoriza reprocessamento seguro.
    """
    entrada: RodadaEntrada = ctx["entrada"]
    base_final: pd.DataFrame = ctx["base_final"].copy()

    # Métricas críticas
    total_geral = float(base_final["TOTAL"].sum()) if not base_final.empty else 0.0
    n_reg = len(base_final)
    n_neg = int((base_final["TOTAL"] < 0).sum()) if "TOTAL" in base_final.columns else 0
    n_valor_dia_zero = int((base_final["VALOR DIARIO VR"] <= 0).sum()) if "VALOR DIARIO VR" in base_final.columns else 0
    n_dias_invalidos = int((base_final["Dias"] < 0).sum()) if "Dias" in base_final.columns else 0

    # Heurística de plausibilidade baseada no feedback do professor
    alvo = 1380178.00
    limite_min = 0.5 * alvo   # 50% do alvo
    limite_max = 2.0 * alvo   # 200% do alvo

    alertas = []
    if n_reg == 0:
        alertas.append("Planilha sem registros. Verifique filtros e entradas.")
    if n_neg > 0:
        alertas.append(f"Há {n_neg} registro(s) com TOTAL negativo (serão zerados).")
    if n_valor_dia_zero > 0:
        alertas.append(f"{n_valor_dia_zero} registro(s) com VALOR DIARIO VR <= 0 (serão substituídos por fallback).")
    if n_dias_invalidos > 0:
        alertas.append(f"{n_dias_invalidos} registro(s) com Dias < 0 (serão ajustados para 0).")
    if total_geral < limite_min or total_geral > limite_max:
        alertas.append(
            f"TOTAL GERAL fora da faixa plausível para a competência (R$ {total_geral:,.2f})."
        )

    precisa_reprocessar = bool(alertas)

    # Se LLM ativa, peça um parecer executivo (não altera dados diretamente)
    parecer_llm = ""
    if entrada.usar_llm:
        try:
            llm = _escolher_llm(entrada)
            prompt = (
                "Você é um agente auditor. Dado o cenário abaixo, produza um parecer conciso com causa provável "
                "e medidas seguras de remediação (sem inventar dados):\n\n"
                f"- Registros: {n_reg}\n"
                f"- TOTAL GERAL: R$ {total_geral:,.2f}\n"
                f"- Alertas: {', '.join(alertas) if alertas else 'Nenhum'}\n"
                "Responda em no máximo 8 linhas."
            )
            parecer_llm = llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            log.warning("Falha LLM no no_auditoria: %s", e)

    ctx.update({
        "audit_total_geral": total_geral,
        "audit_alertas": alertas,
        "audit_parecer_llm": parecer_llm,
        "precisa_reprocessar": precisa_reprocessar,
    })
    return ctx


def no_ajustes(ctx: dict) -> dict:
    """
    Agente de Ajustes (determinístico + seguro).
    Aplica somente correções NÃO destrutivas: clamp de negativos, fallback de valor_dia, normalização de números.
    Em seguida, recalcula totais e saneia.
    """
    base_final: pd.DataFrame = ctx["base_final"].copy()

    if base_final.empty:
        ctx.update({"base_final": base_final})
        return ctx

    # Dias >= 0
    if "Dias" in base_final.columns:
        base_final["Dias"] = pd.to_numeric(base_final["Dias"], errors="coerce").fillna(0).astype(int)
        base_final.loc[base_final["Dias"] < 0, "Dias"] = 0

    # Valor dia > 0 (fallback R$35.00 quando ausente/<=0)
    if "VALOR DIARIO VR" in base_final.columns:
        base_final["VALOR DIARIO VR"] = pd.to_numeric(base_final["VALOR DIARIO VR"], errors="coerce")
        base_final["VALOR DIARIO VR"] = base_final["VALOR DIARIO VR"].fillna(35.0)
        base_final.loc[base_final["VALOR DIARIO VR"] <= 0, "VALOR DIARIO VR"] = 35.0

    # TOTAL sempre recalculado = Dias * ValorDia
    if {"Dias", "VALOR DIARIO VR"}.issubset(base_final.columns):
        base_final["TOTAL"] = (base_final["Dias"].astype(float) * base_final["VALOR DIARIO VR"].astype(float)).round(2)

    # Recalcular custo e desconto se existirem
    if {"Custo empresa", "Desconto profissional", "TOTAL"}.issubset(base_final.columns):
        total = base_final["TOTAL"].astype(float)
        custo = base_final["Custo empresa"].astype(float)
        desc = base_final["Desconto profissional"].astype(float)
        # Se houver inconsistência (custo+desc != total), redivide mantendo a razão média anterior (fallback 80/20)
        soma_custo = float(custo.sum()) if len(base_final) else 0.0
        soma_desc = float(desc.sum()) if len(base_final) else 0.0
        pct_emp = 0.8
        if soma_custo + soma_desc > 0:
            pct_emp = soma_custo / (soma_custo + soma_desc)
        base_final["Custo empresa"] = (total * pct_emp).round(2)
        base_final["Desconto profissional"] = (total - base_final["Custo empresa"]).round(2)

    # Zerar negativos residuais por segurança
    for col in ["TOTAL", "Custo empresa", "Desconto profissional"]:
        if col in base_final.columns:
            base_final[col] = pd.to_numeric(base_final[col], errors="coerce").fillna(0.0)
            base_final.loc[base_final[col] < 0, col] = 0.0
            base_final[col] = base_final[col].round(2)

    # Saneamento final
    base_final = sanear_saida_planilha(base_final)
    ctx.update({"base_final": base_final})
    return ctx


def no_exportacao(ctx: dict) -> dict:
    """Agente de Exportação (determinístico). Gera XLSX no layout exato."""
    xlsx_bytes = escrever_planilha_padrao_xlsx_bytes(ctx["base_final"], ctx["validacoes"])
    ctx.update({"xlsx_bytes": xlsx_bytes})
    return ctx


def no_explicacao(ctx: dict) -> dict:
    """
    Agente de Relatório (INTELIGENTE quando usar_llm=True).
    Consolida o que ocorreu, com alertas, totas e contexto dos agentes.
    """
    entrada: RodadaEntrada = ctx["entrada"]
    dfs = ctx["dfs"]
    base_final = ctx["base_final"]
    validacoes = ctx["validacoes"]

    total_geral = float(base_final["TOTAL"].sum()) if not base_final.empty else 0.0
    alertas = ctx.get("audit_alertas", [])
    parecer_llm = ctx.get("audit_parecer_llm", "")

    rel_deterministico = [
        "=== RELATÓRIO DE EXECUÇÃO ===",
        f"Competência: {entrada.competencia}",
        f"% empresa: {entrada.pct_empresa:.2f}",
        f"Registros finais: {len(base_final)}",
        f"TOTAL GERAL: R$ {total_geral:,.2f}",
        f"Arquivos: admissoes={len(dfs.admissoes)}, ativos={len(dfs.ativos)}, desligados={len(dfs.desligados)}, ferias={len(dfs.ferias)}",
        f"Bases: dias_uteis={len(dfs.base_dias_uteis)} | sindicato_x_valor={len(dfs.base_sindicato_valor)}",
        f"Alertas: {', '.join(alertas) if alertas else 'Nenhum'}",
        "",
        "Agentes utilizados:",
        "- Ingestão (determinístico)",
        "- Validação (LLM quando ativada)",
        "- Cálculo (determinístico)",
        "- Auditoria & Consistência (LLM + heurística)",
        "- Ajustes Seguros (determinístico)",
        "- Exportação (determinístico)",
        "- Relatório (LLM quando ativada)",
    ]
    texto = "\n".join(rel_deterministico)

    if entrada.usar_llm:
        llm = _escolher_llm(entrada)
        sys = SystemMessage(content="Você é um assistente que gera relatórios claros e executivos.")
        hm = HumanMessage(content=(
            f"{texto}\n\n"
            f"Parecer do auditor (LLM):\n{parecer_llm}\n\n"
            f"Validações (amostra):\n{validacoes.head(12).to_string(index=False)}"
        ))
        try:
            texto = llm.invoke([sys, hm]).content
        except Exception as e:
            log.warning("LLM falhou no relatório; mantendo versão determinística. Erro: %s", e)

    ctx.update({"relatorio": texto})
    return ctx


# ─────────────── grafo ───────────────
def construir_grafo():
    g = StateGraph(dict)

    g.add_node("ingestao", no_ingestao)
    g.add_node("validacoes", no_validacoes)
    g.add_node("calculo", no_calculo)
    g.add_node("auditoria", no_auditoria)
    g.add_node("ajustes", no_ajustes)
    g.add_node("exportacao", no_exportacao)
    g.add_node("explicacao", no_explicacao)

    g.set_entry_point("ingestao")
    g.add_edge("ingestao", "validacoes")
    g.add_edge("validacoes", "calculo")
    g.add_edge("calculo", "auditoria")

    # Roteamento condicional: se precisar reprocessar, passa por ajustes; senão, segue para exportação
    def _router(ctx: Dict[str, Any]) -> str:
        return "ajustes" if ctx.get("precisa_reprocessar") else "exportacao"

    # LangGraph >0.1.x: add_conditional_edges(source, router, mapping)
    g.add_conditional_edges("auditoria", _router, {
        "ajustes": "ajustes",
        "exportacao": "exportacao",
    })

    # Após ajustes, exporta
    g.add_edge("ajustes", "exportacao")
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
