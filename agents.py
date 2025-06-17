# agents.py

from __future__ import annotations

import logging
import os
import re
import sqlite3
import textwrap
import unicodedata
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import SQLDatabase

# ────────────────────────── logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("CsvAgent")

# ────────────────────────── helpers ─────────────────────────
_re_bad = re.compile(r"[^0-9a-zA-Z_]+")


def _snake(text: str) -> str:
    """Converte para snake_case ASCII-safe."""
    return _re_bad.sub(
        "_", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    ).strip("_").lower()


def _get_llm():
    """Retorna o LLM baseado na configuração do usuário."""
    model_provider = st.session_state.get("model_provider", "OpenAI")
    model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
    
    if model_provider == "OpenAI":
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=st.session_state.get("OPENAI_API_KEY")
        )
    elif model_provider == "Ollama":
        return ChatOllama(
            model=model_name,
            base_url=st.session_state.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0
        )
    elif model_provider == "Google Gemini":
        if model_name == "gemini-1.5-flash":
            model_name = "models/gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=st.session_state.get("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(f"Provedor de modelo não suportado: {model_provider}")


# ────────────────────────── agente ──────────────────────────
class CsvAgent:
    """Carrega CSVs → SQLite e responde perguntas em linguagem natural."""

    # -------------------------------------------------------- #
    def __init__(self, csv_paths: List[str], db_path: str = "nfs.db"):
        self.llm = _get_llm()
        self.db_path = db_path
        self._csvs_to_sqlite(csv_paths)
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        self._build_prompts()

    # ---------------- CSV → SQLite ----------------
    def _csvs_to_sqlite(self, paths: List[str]):
        con = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        self.tables: Dict[str, Dict] = {}
        for p in paths:
            tname = _snake(Path(p).stem)
            try:
                df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8")
            except Exception as err:
                log.error("Falha ao ler %s: %s", p, err)
                continue
            df.columns = [_snake(c) for c in df.columns]
            df.to_sql(tname, con, if_exists="replace", index=False)
            self.tables[tname] = {"rows": len(df), "columns": list(df.columns)}
            log.info("✓ %s → %s (%d linhas)", Path(p).name, tname, len(df))
        con.close()
        if not self.tables:
            raise RuntimeError("Nenhum CSV válido encontrado.")

    # ---------------- Prompts ----------------
    def _build_prompts(self):
        schema = "\n".join(
            f"- {t} ({', '.join(d['columns'])}) | {d['rows']} linhas"
            for t, d in self.tables.items()
        )

        self.sys_sql = SystemMessage(
            content=textwrap.dedent(
                f"""
                Você é um assistente SQL com acesso a um banco SQLite.

                ESQUEMA:
                {schema}

                Regras:
                1. Gere apenas o comando SQL (sem ``` e sem comentários).
                2. Teste com EXPLAIN; se der erro, corrija antes de responder.
                """
            )
        )

        self.sys_explain = SystemMessage(
            content=textwrap.dedent(
                """
                Explique os resultados em português.
                Formate dinheiro como R$ X,XX e datas DD/MM/AAAA.
                Não inclua o SQL.
                """
            )
        )

    # ---------------- limpeza & quoting ----------------
    _fence = re.compile(r"```(?:sql)?|```", re.I)
    _needs_quote = re.compile(r"\b\d\w*")  # token que começa por número

    @classmethod
    def _clean_sql(cls, raw: str) -> str:
        # 1) remove cercas  ```  e 2) remove todos os back-ticks `
        sql = cls._fence.sub("", raw).replace("`", "").strip()
        # mantém só até o primeiro ';'
        if ";" in sql:
            sql = sql.split(";", 1)[0] + ";"
        # adiciona aspas duplas se começar por dígito
        sql = cls._needs_quote.sub(lambda m: f'"{m.group(0)}"', sql)
        return sql.strip()

    # ---------------- execução ----------------
    def _generate_sql(self, question: str) -> str:
        try:
            resp = self.llm([self.sys_sql, HumanMessage(content=question)])
        except ResourceExhausted:
            self.llm = _get_llm()
            resp = self.llm([self.sys_sql, HumanMessage(content=question)])
        return self._clean_sql(resp.content)

    def _run_sql(self, sql_cmd: str) -> str | None:
        try:
            return self.db.run(sql_cmd)
        except Exception as err:
            log.warning("Tentativa SQL falhou: %s", err)
            return None

    # ---------------- interface pública ----------------
    def invoke(self, question: str) -> str:
        log.info("Pergunta: %s", question)

        sql_cmd = self._generate_sql(question)
        result = self._run_sql(sql_cmd)

        # tenta corrigir uma vez se falhar
        if result is None:
            correction_prompt = (
                f"O SQLite retornou erro ao executar:\n{sql_cmd}\n"
                "Corrija o comando SQL."
            )
            sql_cmd2 = self._clean_sql(
                self.llm(
                    [self.sys_sql, HumanMessage(content=correction_prompt)]
                ).content
            )
            result = self._run_sql(sql_cmd2)
            if result is None:
                return "❌ SQL gerado inválido mesmo após correção."

        explanation = self.llm(
            [
                self.sys_explain,
                HumanMessage(content=f"Pergunta: {question}\nResultado:\n{result}"),
            ]
        ).content
        return explanation