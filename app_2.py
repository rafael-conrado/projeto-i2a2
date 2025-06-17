# app.py

from __future__ import annotations
import io, zipfile, tempfile
from pathlib import Path
from typing import List

import streamlit as st
from agents import CsvAgent

# imagem no cabeçalho
image_url = "https://github.com/rafael-conrado/projeto-i2a2/blob/main/imagem/orbis.png"
st.image(image_url, use_column_width=True)

# Nome do aplicativo
st.set_page_config(page_title="Agente NF Analytics", page_icon="🌀")
st.title("📊 Agente NF Analytics ")

# Seleção do modelo
model_provider = st.selectbox(
    "Escolha o provedor do modelo",
    ["OpenAI", "Ollama", "Google Gemini"]
)
st.session_state["model_provider"] = model_provider

# Configuração específica para cada provedor
if model_provider == "OpenAI":
    openaikey = st.text_input("Enter your OpenAI API Key", type='password')
    if openaikey:
        st.session_state["OPENAI_API_KEY"] = openaikey
    model_name = st.selectbox(
        "Escolha o modelo OpenAI",
        ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"]
    )
    st.session_state["model_name"] = model_name
elif model_provider == "Ollama":
    model_name = st.selectbox(
        "Escolha o modelo Ollama",
        ["llama2", "mistral", "codellama", "neural-chat"]
    )
    st.session_state["model_name"] = model_name
    ollama_base_url = st.text_input("Ollama Base URL (opcional)", value="http://localhost:11434")
    st.session_state["OLLAMA_BASE_URL"] = ollama_base_url
elif model_provider == "Google Gemini":
    google_api_key = st.text_input("Enter your Google API Key", type='password')
    if google_api_key:
        st.session_state["GOOGLE_API_KEY"] = google_api_key
    model_name = st.selectbox(
        "Escolha o modelo Gemini",
        ["gemini-pro", "gemini-pro-vision", "gemini-1.5-flash"]
    )
    st.session_state["model_name"] = model_name

# ---------------- upload -----------------
files = st.file_uploader(
    "Arraste CSVs ou ZIP", type=["csv", "zip"], accept_multiple_files=True
)

def _save_tmp(uploaded) -> List[str]:
    tmpdir = tempfile.mkdtemp(prefix="csv_agent_")
    out: list[str] = []
    for f in uploaded:
        if f.name.lower().endswith(".csv"):
            p = Path(tmpdir) / f.name
            p.write_bytes(f.getbuffer())
            out.append(str(p))
        elif f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getbuffer())) as z:
                for m in z.namelist():
                    if m.lower().endswith(".csv"):
                        p = Path(tmpdir) / Path(m).name
                        p.write_bytes(z.read(m))
                        out.append(str(p))
    return out

if files:
    csv_paths = _save_tmp(files)
    if not csv_paths:
        st.error("Nenhum CSV encontrado nos arquivos enviados.")
        st.stop()

    @st.cache_resource(show_spinner="🔧 Construindo agente…")
    def _cached(paths_key: tuple[str]):
        return CsvAgent(list(paths_key))

    agent = _cached(tuple(sorted(csv_paths)))

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Pergunte algo sobre os CSVs…")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"), st.spinner("⏳ Processando…"):
            answer = agent.invoke(question)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})
else:
    st.info("Carregue um ou mais CSVs (ou ZIP) para começar.")
