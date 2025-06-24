🧠 Projeto I2A2

Sistema de análise inteligente de documentos com foco em LLMs, agentes autônomos, e integração via n8n, permitindo automação de fluxos com NLP, análise de PDFs, e APIs de IA.


---

⚙️ Tecnologias Utilizadas

Python 3.10+

FastAPI + Uvicorn

LlamaIndex

LangChain

OpenAI / HuggingFace LLMs

n8n (Automação de workflows)

Docker + Docker Compose

PyMuPDF (para leitura de PDFs)

dotenv (configuração via .env)



---

📁 Estrutura do Projeto

projeto-i2a2/
├── api/                # Endpoints FastAPI
├── app/                # Lógica da aplicação
│   ├── agents/         # Agentes de recuperação LangChain
│   ├── core/           # Configuração (logger, env)
│   ├── documents/      # Ingestão e extração de PDFs
│   └── services/       # Serviços auxiliares
├── scripts/            # Scripts de ingestão e consulta
├── n8n/                # Configuração do n8n e workflows
│   └── workflows/      # Workflows salvos/exportados
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── main.py             # Ponto de entrada FastAPI


---

🚀 Como Rodar o Projeto

🐳 Com Docker Compose

git clone https://github.com/rafael-conrado/projeto-i2a2.git
cd projeto-i2a2
cp .env.example .env
docker-compose up --build

Serviços disponíveis:

FastAPI: http://localhost:8000/docs

n8n: http://localhost:5678



---

💻 Localmente (sem Docker)

1. Crie um ambiente virtual:



python -m venv venv
source venv/bin/activate

2. Instale as dependências:



pip install -r requirements.txt

3. Rode a API:



uvicorn main:app --reload


---

🔄 Sobre o n8n

O projeto utiliza o n8n para:

Automatizar fluxos de ingestão e consulta de documentos

Integrar com APIs externas (ex: GPT, webhook, email, banco de dados)

Orquestrar rotinas com base em eventos


📥 Acessando o n8n

Com Docker, acesse o painel do n8n:

http://localhost:5678

Use as credenciais padrão (ou configure via .env).

📂 Workflows

Os workflows estão salvos na pasta:

n8n/workflows/

Você pode importar no painel do n8n ou apontar o volume no docker-compose.yml para manter persistência.


---

📚 Funcionalidades

📄 Ingestão de PDFs: Faz parsing e indexação via LlamaIndex

❓ Consulta semântica: Respostas inteligentes com LLMs

🧠 Agentes modulares: Criação de agentes para diferentes domínios

🔄 Automação com n8n: dispara fluxos automáticos em tempo real

📡 API REST: Disponibilização dos recursos via endpoints FastAPI



---

📌 Endpoints Principais

POST /ingest — Envia documento para ser analisado

POST /query — Envia uma pergunta ao sistema

GET /health — Status da aplicação


Acesse: http://localhost:8000/docs


---

✅ Pré-requisitos

OpenAI API Key (ou outro modelo)

Docker + Docker Compose

.env corretamente configurado (ver .env.example)

