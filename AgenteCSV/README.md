# 📊 Agente NF Analytics

Um aplicativo Streamlit que permite a análise de arquivos CSV (incluindo aqueles compactados em ZIP) usando Modelos de Linguagem Grande (LLMs) para responder a perguntas em linguagem natural.

## 🌟 O que é?

O "Agente NF Analytics" é uma ferramenta interativa construída com Streamlit que atua como um "agente inteligente" para seus dados em CSV. Ele permite que você carregue um ou múltiplos arquivos CSV (ou um arquivo ZIP contendo CSVs) e, em seguida, faça perguntas sobre esses dados em português. Nos bastidores, o aplicativo utiliza um LLM para converter suas perguntas em comandos SQL, executar esses comandos em um banco de dados SQLite (criado temporariamente a partir de seus CSVs) e, finalmente, interpretar os resultados de volta para uma explicação compreensível.

## ⚙️ Como Funciona?

1.  **Upload de Arquivos**: Você faz o upload de um ou mais arquivos CSV, ou um arquivo ZIP contendo CSVs.
2.  **Processamento Interno**:
    * Os arquivos CSV são lidos e suas colunas são "sanitizadas" para um formato seguro para banco de dados (snake_case).
    * Cada CSV é então carregado em uma tabela separada dentro de um banco de dados SQLite temporário.
    * Um agente LLM é inicializado, configurado com o esquema do banco de dados (nomes das tabelas e colunas).
3.  **Interação com o Usuário**:
    * Você seleciona o provedor do modelo (OpenAI, Ollama, Google Gemini) e o modelo específico que deseja usar.
    * Insere sua chave de API, se necessário, para os provedores de modelos pagos.
    * Você digita sua pergunta em linguagem natural sobre os dados carregados.
4.  **Geração e Execução de SQL**:
    * O agente LLM recebe sua pergunta e o esquema do banco de dados.
    * Ele gera um comando SQL apropriado para responder à sua pergunta.
    * Este comando SQL é executado no banco de dados SQLite.
    * Se houver um erro na execução do SQL, o agente tenta corrigi-lo e executa novamente.
5.  **Geração da Resposta**:
    * Os resultados da consulta SQL são passados de volta para o LLM.
    * O LLM interpreta esses resultados e os traduz para uma explicação clara e formatada em português, apresentando-os a você.

## 🚀 Como Testar (Rodar Localmente)

Siga os passos abaixo para configurar e rodar o aplicativo em sua máquina local:

1.  **Clone o Repositório** (se ainda não o fez):
    ```bash
    git clone https://github.com/rafael-conrado/projeto-i2a2
    cd projeto-i2a2
    ```

2.  **Crie e Ative um Ambiente Virtual** (recomendado):
    ```bash
    python -m venv venv
    # No Windows:
    .\venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instale as Dependências**:
    Certifique-se de que você tem o `requirements.txt` no diretório raiz do projeto.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare suas Chaves de API** (se for usar OpenAI ou Google Gemini):
    * **OpenAI**: Obtenha sua API Key em [OpenAI Platform](https://platform.openai.com/).
    * **Google Gemini**: Obtenha sua API Key em [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Você precisará inserir estas chaves no campo apropriado dentro do aplicativo Streamlit quando ele estiver em execução.

5.  **Execute o Aplicativo Streamlit**:
    ```bash
    streamlit run app.py
    ```

    Isso abrirá o aplicativo no seu navegador padrão.

6.  **Interaja com o Aplicativo**:
    * No navegador, selecione o provedor do modelo e o modelo desejado.
    * Insira sua chave de API, se aplicável.
    * Clique em "Arraste CSVs ou ZIP" para fazer o upload dos seus arquivos de dados.
    * Uma vez que os arquivos são processados, uma caixa de chat aparecerá.
    * Comece a fazer perguntas sobre seus dados! Por exemplo: "Qual o total de vendas por produto?", "Quais clientes compraram mais de R$ 1000?", etc.

## ✨ Para que Serve?

O "Agente NF Analytics" é ideal para:

* **Análise Rápida de Dados**: Obtenha insights de seus arquivos CSV sem a necessidade de escrever consultas SQL complexas ou scripts de programação.
* **Usuários Não Técnicos**: Permite que pessoas sem conhecimento em SQL ou programação analisem grandes volumes de dados.
* **Exploração de Dados**: Facilita a exploração e a descoberta de padrões em seus conjuntos de dados de forma conversacional.
* **Relatórios Ad-hoc**: Gere relatórios e resumos rápidos sobre seus dados para tomadas de decisão.
* **Análise de Notas Fiscais (NF)**: Embora genérico para CSVs, o nome "NF Analytics" sugere uma aplicação específica para dados de notas fiscais, permitindo perguntas como "Qual o valor total das notas fiscais do mês passado?", "Quais produtos foram mais vendidos na região X?", etc.
* **Prototipagem Rápida**: Desenvolva e teste rapidamente modelos de linguagem em diferentes plataformas (OpenAI, Ollama, Gemini) para suas necessidades de análise de dados.

Com esta ferramenta, a análise de dados se torna tão simples quanto fazer uma pergunta!