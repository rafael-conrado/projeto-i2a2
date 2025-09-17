# 🍽️ Automação de Compra de VR/VA

Projeto para automatizar o cálculo e geração de planilhas de compra de Vale-Refeição (VR) e Vale-Alimentação (VA), aplicando regras de negócio exigidas em aula e validando com agentes inteligentes (LLMs).

---

## 📂 Estrutura do Projeto

```
AgenteVR/
├── app.py # Interface Streamlit
├── agentes.py # Orquestração de agentes
├── vrva_funcoes.py # Funções de domínio e cálculos
├── requirements.txt # Dependências
└── README.md # Este arquivo

```
---

## 📥 Entradas Necessárias

As seguintes planilhas (.xlsx ou .xls) devem ser fornecidas, juntas ou em um único **ZIP**:

- `ADMISSÃO ABRIL.xlsx` — admissões recentes.  
- `ATIVOS.xlsx` — colaboradores ativos.  
- `DESLIGADOS.xlsx` — desligamentos e comunicados.  
- `FÉRIAS.xlsx` — férias registradas.  
- `AFASTAMENTOS.xlsx` — afastamentos/licenças.  
- `EXTERIOR.xlsx` — valores especiais (overrides).  
- `APRENDIZ.xlsx` — lista de aprendizes.  
- `ESTÁGIO.xlsx` — lista de estagiários.  
- `Base dias uteis.xlsx` — número de dias úteis por sindicato.  
- `Base sindicato x valor.xlsx` — valor diário por estado/sindicato.  

---

## 📤 Saídas Geradas

- `VR MENSAL 05.2025.xlsx` → planilha final no padrão exigido.  
- `VRVA_base_final_05-2025.csv` → versão em CSV.  
- `VRVA_validacoes_05-2025.csv` → validações em CSV.  

---

## ▶️ Como Executar

### **1. Crie e ative um ambiente virtual (opcional, mas recomendado):**

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```
### **2. Instale as dependências:**

```bash
pip install -r requirements.txt
```
### **3. Rodar aplicação Streamlit**

```bash
streamlit run app.py
```

### **4. Configurar execução:**

- Selecionar a competência (ex.: 2025-05).
- Definir % empresa (ex.: 0.80 para 80/20).
- Ativar ou não o uso da LLM.
- Fazer upload dos arquivos (ZIP ou planilhas).

---

## ✅ Checklist de Conformidade

- [x] Nome do arquivo final **VR MENSAL 05.2025.xlsx**  
- [x] Layout respeitando as 10 colunas originais  
- [x] Validações idênticas ao modelo do professor  
- [x] Tratamento de aprendizes, estagiários e diretores  
- [x] Proporcionalidade em férias, admissões e desligamentos  
- [x] Suporte a afastamentos e exteriores  
- [x] Uso explícito de agentes inteligentes quando LLM ativo  
- [x] Relatório final gerado automaticamente 

---

## 🤖 Uso de Agentes Inteligentes

O sistema utiliza agentes organizados em um grafo de execução. Eles se dividem em **determinísticos** e **inteligentes (com LLM)**:

### 🔹 Determinísticos
- **Ingestão** → Extrai ZIP e carrega planilhas  
- **Cálculo** → Aplica regras de negócio e gera base final  
- **Ajustes Seguros** → Corrige valores inválidos (negativos, zeros, etc.)  
- **Exportação** → Gera a planilha no layout exigido  

### 🔹 Inteligentes (LLM, quando ativado)
- **Validação** → Identifica problemas estruturais nas bases e sugere correções  
- **Auditoria & Consistência** → Analisa plausibilidade dos totais e emite parecer executivo  
- **Relatório** → Gera relatório consolidado em linguagem clara e executiva  

> Caso a opção **LLM esteja desativada**, todos os agentes executam apenas suas regras determinísticas.

---

## 📝 Relatório Automático

Ao final da execução, o sistema gera um relatório consolidado contendo:

- Competência processada  
- Percentual empresa/profissional aplicado  
- Quantidade de registros finais  
- Total geral calculado  
- Quantidade de registros em cada planilha base (ativos, desligados, férias etc.)  
- Alertas detectados (ex.: valores negativos, totais fora da faixa plausível)  
- Indicação dos agentes utilizados  
- (Quando LLM ativado) parecer executivo da auditoria com causas prováveis e recomendações  

Esse relatório é exibido diretamente na interface do Streamlit.

---

## 📄 Licença e Créditos

Este projeto foi desenvolvido como parte do **Desafio 4** do curso de agentes inteligentes.  
Todos os direitos sobre os dados utilizados (planilhas de RH) pertencem exclusivamente à organização fornecedora e foram utilizados apenas para fins acadêmicos.

### 👥 Equipe
- **Nome do Grupo:** Aurora Digital  
- **Integrantes:**
  - Carolina Prado  
  - Claudia Welter  
  - José Rodriguez  
  - Rafael Conrado  
  - Ricardo Hamada  
  - Yldiane de Carvalho  

### 📌 Observação
O código-fonte é de uso acadêmico e pode ser reutilizado/expandido para estudos, desde que mantidos os créditos ao grupo **Aurora Digital** e ao professor que forneceu o enunciado do desafio.