# app.py

from __future__ import annotations
import io
import os
import zipfile
import logging
from typing import List, Optional, Dict
import streamlit as st
from dotenv import load_dotenv
from agentes import construir_grafo, RodadaEntrada
from vrva_funcoes import df_para_streamlit

# ─────────────────── logging ───────────────────
log = logging.getLogger("vrva.app")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

load_dotenv()

st.set_page_config(page_title="Automação de Compra de VR/VA", page_icon="🍽️", layout="centered")
st.title("🍽️ Automação de Compra de VR/VA")

# ─────────────────── util UI ───────────────────
PROVIDER_MODELS: Dict[str, list[str]] = {
    "gemini": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
        "Outro (digitar…)",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
        "Outro (digitar…)",
    ],
    "openrouter": [
        "deepseek/deepseek-r1-0528",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-1.5-pro",
        "openai/gpt-4o-mini",
        "Outro (digitar…)",
    ],
}

def empacotar_zip_em_memoria(
    arquivos: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> Optional[bytes]:
    if not arquivos:
        return None
    # prioriza um ZIP enviado
    for f in arquivos:
        if f.name.lower().endswith(".zip"):
            return f.getvalue()
    # caso contrário, empacota XLSX/XLS em um ZIP em memória
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in arquivos:
            if f.name.lower().endswith((".xlsx", ".xls")):
                zf.writestr(f.name, f.getvalue())
    return mem.getvalue()

# ─────────────────── parâmetros ───────────────────
c1, c2, c3 = st.columns(3)
with c1:
    competencia = st.text_input(
        "Competência", value="2025-05",
        help="Use AAAA-MM ou MM/AAAA (ex.: 2025-05 ou 05/2025)."
    )
with c2:
    pct_empresa = st.number_input(
        "% empresa (0.0–1.0)", min_value=0.0, max_value=1.0, value=0.80, step=0.05,
        help="Fração do benefício paga pela empresa. Ex.: 0,80 = 80% empresa / 20% profissional."
    )
with c3:
    usar_llm = st.toggle(
        "Ativar LLM (agente supervisor)", value=True,
        help="Quando ativo, a LLM supervisiona validações, explicações e fallbacks."
    )

# Bloco LLM só aparece quando usar_llm=True
modelo_escolhido: Optional[str] = None
provedor = None
gemini_key = openai_key = openrouter_key = None

with st.container():
    st.markdown("#### Modelos e chaves de API")
    if usar_llm:
        c4, c5 = st.columns(2)
        with c4:
            provedor = st.selectbox("Provedor LLM", ["gemini", "openai", "openrouter"], index=0)
        with c5:
            opcoes = PROVIDER_MODELS.get(provedor, ["Outro (digitar…)"])
            modelo_sel = st.selectbox("Modelo", opcoes, index=0)
        if modelo_sel == "Outro (digitar…)":
            modelo_escolhido = st.text_input(
                "Modelo (custom)",
                value="",
                placeholder="Ex.: gemini-1.5-flash / gpt-4o-mini / deepseek/deepseek-r1-0528",
            ).strip() or None
        else:
            modelo_escolhido = modelo_sel

        st.caption("Informe a chave de API do provedor selecionado (ou configure via variáveis de ambiente).")
        if provedor == "gemini":
            gemini_key = st.text_input(
                "GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY", "")
            )
        elif provedor == "openai":
            openai_key = st.text_input(
                "OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", "")
            )
        elif provedor == "openrouter":
            openrouter_key = st.text_input(
                "OPENROUTER_API_KEY", type="password", value=os.getenv("OPENROUTER_API_KEY", "")
            )
    else:
        st.info("LLM desativada: o pipeline executará apenas as regras determinísticas.")

st.markdown("#### Envie o pacote `.zip` **ou** as planilhas (.xlsx/.xls)")
uploads = st.file_uploader("Arraste aqui", type=["zip", "xlsx", "xls"], accept_multiple_files=True)

# ─────────────────── execução ───────────────────
if uploads:
    pacote_zip = empacotar_zip_em_memoria(uploads)
    if not pacote_zip:
        st.error("Nenhum arquivo válido foi enviado.")
        st.stop()

    # Valida chave quando LLM estiver ativa:
    if usar_llm:
        if provedor == "gemini" and not (gemini_key or os.getenv("GEMINI_API_KEY")):
            st.error("Informe GEMINI_API_KEY para usar Gemini.")
            st.stop()
        if provedor == "openai" and not (openai_key or os.getenv("OPENAI_API_KEY")):
            st.error("Informe OPENAI_API_KEY para usar OpenAI.")
            st.stop()
        if provedor == "openrouter" and not (openrouter_key or os.getenv("OPENROUTER_API_KEY")):
            st.error("Informe OPENROUTER_API_KEY para usar OpenRouter.")
            st.stop()

    entrada = RodadaEntrada(
        zip_bytes=pacote_zip,
        competencia=competencia.strip(),
        pct_empresa=float(pct_empresa),
        usar_llm=bool(usar_llm),
        provedor=(provedor or "gemini").strip(),
        modelo=modelo_escolhido or None,
        gemini_key=(gemini_key or os.getenv("GEMINI_API_KEY") or None),
        openai_key=(openai_key or os.getenv("OPENAI_API_KEY") or None),
        openrouter_key=(openrouter_key or os.getenv("OPENROUTER_API_KEY") or None),
    )

    try:
        with st.spinner("🧠 Orquestrando agentes…"):
            grafo = construir_grafo()
            saida = grafo.invoke(entrada)

        st.success("✅ Processamento concluído.")
        st.caption(f"Arquivos extraídos em: {saida.tmpdir}")

        st.markdown("### 📋 Relatório (Agente de Explicação)")
        st.code(saida.resultado.relatorio, language="text")

        st.markdown("### ✅ Pré‑visualização (100 primeiras linhas)")
        st.dataframe(df_para_streamlit(saida.resultado.base_final.head(100)), use_container_width=True)

        st.markdown("### 🛡️ Validações")
        st.dataframe(df_para_streamlit(saida.resultado.validacoes), use_container_width=True)

        @st.cache_data
        def _to_csv_bytes(df):
            return df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Baixar base final (CSV)",
            data=_to_csv_bytes(saida.resultado.base_final),
            file_name=f"VRVA_base_final_{entrada.competencia.replace('/','-')}.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Baixar validações (CSV)",
            data=_to_csv_bytes(saida.resultado.validacoes),
            file_name=f"VRVA_validacoes_{entrada.competencia.replace('/','-')}.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Baixar planilha no padrão (XLSX)",
            data=saida.xlsx_bytes,
            file_name=f"VR_MENSAL_{entrada.competencia.replace('/','-')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"❌ Erro no processamento: {e}")
        st.stop()
else:
    st.info("Envie um ZIP **ou** as planilhas (.xlsx/.xls) para iniciar.")
