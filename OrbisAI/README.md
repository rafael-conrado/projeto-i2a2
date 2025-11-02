# Agente Fiscal

Auditoria fiscal pragmática com Python + Streamlit + SQLite e (opcional) suporte a LLM para análises e insights. O projeto orquestra parsing de XML, validações de regras, trilha de auditoria e fluxos de revisão com interface simples e produtiva para times fiscais.

## Principais recursos

- Upload e processamento de documentos fiscais (XML) com identificação de duplicidade e riscos
- Edição de cabeçalho e itens com trilha de auditoria (revisões) e rollback
- Fluxos de aprovação, reprocessamento e revalidação
- Ações em lote sobre o conjunto filtrado (aprovar, revisão, revalidar, exportar CSV)
- Duplicidade: vincular/arquivar e marcar falso positivo
- Métricas básicas com geração de insights (opcional via LLM)
- Administração: usuários, manutenção do banco e editor de regras fiscais (YAML)

## Arquitetura rápida

- UI: Streamlit (`app.py`)
- Banco: SQLite (arquivo local), camada em `banco_de_dados.py`
- Orquestração: `orchestrator.py` chama agentes específicos
- Agentes: parsing de XML, normalização, OCR (desativado), NLP analítico, métricas
- Validação: `validacao.py` lê regras YAML (edição em Administração)
- Memória de sessão: estado cognitivo/blackboard em `memoria.py`

## Requisitos

- Python 3.11 ou 3.12
- Pip e venv (recomendado)

## Instalação

```powershell
# 1) Criar e ativar venv (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# 2) Dependências
pip install -r requirements.txt
```

## Execução

```powershell
# Rodar a aplicação
streamlit run app.py
```

- Navegue para a URL exibida no terminal (geralmente http://localhost:8501).
- No primeiro uso, se o banco estiver vazio, será criado um usuário administrador padrão.
  - Email: `admin@i2a2.academy`
  - Senha: `admin123`
  - Importante: altere a senha após o primeiro login.

## Configuração de LLM (opcional)

Na barra lateral, informe o provedor/modelo e a chave:

- Provedores suportados: `gemini`, `openai`, `openrouter`
- Chave via variável de ambiente (recomendado) ou campo de entrada

```powershell
# Exemplo (PowerShell):
$env:GEMINI_API_KEY = "sua_chave"
$env:OPENAI_API_KEY = "sua_chave"
$env:OPENROUTER_API_KEY = "sua_chave"
```

O modo "Seguro (sem IA)" pode ser ativado/desativado nas Análises. A aplicação roteia automaticamente perguntas complexas para LLM quando disponível.

## Perfis e permissões

- admin: total, incluindo ações em lote e administração
- auditor: edita itens/cabeçalho, revalida
- conferente: edição básica e revalidação
- operador: leitura

## Estrutura de dados (tabelas principais)

- `documentos`: cabeçalho, status, meta_json
- `itens`, `impostos`: detalhamento por item
- `revisoes`: trilha de auditoria (quem, quando, campo antes/depois)
- `usuarios`: autenticação e perfis
- `metricas`, `config`: telemetria e preferências

## Notas de segurança

- As credenciais padrão são apenas para desenvolvimento. Altere a senha do administrador no primeiro login.
- Evite expor chaves de API em arquivo; prefira variáveis de ambiente.

## Troubleshooting

- Erro de IDs/keys duplicadas no Streamlit: geralmente causado por widgets duplicados; cada botão/campo deve ter `key` único.
- Tabelas vazias: verifique se o XML foi processado e se não houve conflito de hash (duplicata).
- LLM não funciona: confirme provedor/modelo e a presença da variável de ambiente correta.

## Estrutura do repositório

```
app.py
banco_de_dados.py
memoria.py
modelos_llm.py
orchestrator.py
regras_fiscais.yaml
requirements.txt
validacao.py
agentes/
  ├─ __init__.py
  ├─ agente_analitico.py
  ├─ agente_associador_xml.py
  ├─ agente_nlp.py
  ├─ agente_normalizador.py
  ├─ agente_ocr.py
  ├─ agente_xml_parser.py
  ├─ metrics_agent.py
  └─ utils.py
```

## Licença

Este projeto é distribuído sob a **MIT License** — veja [LICENSE](LICENSE). Você pode usar, copiar, modificar e distribuir livremente, desde que mantenha o aviso de copyright e a licença.
