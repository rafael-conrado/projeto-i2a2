# 🍽️ Automação de Compra de VR/VA

Sistema inteligente de automação para processamento de benefícios de Vale Refeição (VR) e Vale Alimentação (VA) com supervisão de IA.

## 📋 Sobre o Projeto

Este sistema automatiza o processo de cálculo e geração de relatórios de benefícios alimentação para funcionários, utilizando **agentes inteligentes** para validação e supervisão dos dados. O projeto combina automação de processos com inteligência artificial para garantir precisão e eficiência na gestão de benefícios corporativos.

### ✨ Principais Funcionalidades

- 🔄 **Processamento Automático** de planilhas de funcionários
- 🤖 **Supervisão Inteligente** com LLMs (Gemini, OpenAI, OpenRouter)
- 📊 **Detecção Automática** de tipos de planilha
- ✅ **Validações Automáticas** de integridade de dados
- 🧮 **Cálculos Complexos** com regras de negócio
- 📈 **Geração de Relatórios** executivos
- 📥 **Interface Web** intuitiva com Streamlit

## 🏗️ Arquitetura

O sistema é construído com uma arquitetura de **agentes orquestrados**:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Ingestão  │───▶│ Validações  │───▶│  Cálculo    │───▶│ Exportação  │───▶│ Explicação  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │                   │
   Extrai ZIP         Verifica dados      Processa regras      Gera XLSX        Relatório IA
   Carrega dados      Valida integridade  Calcula benefícios   Formata planilha  Explica resultados
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- pip

### Passos de Instalação

1. **Clone o repositório**
```bash
git clone https://github.com/rafael-conrado/projeto-i2a2
cd projeto-i2a2
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Configure as variáveis de ambiente** (opcional)
```bash
# Crie um arquivo .env
GEMINI_API_KEY=sua_chave_gemini
OPENAI_API_KEY=sua_chave_openai
OPENROUTER_API_KEY=sua_chave_openrouter
```

## 🎯 Como Usar

### 1. Preparação dos Dados

Prepare um arquivo ZIP contendo as seguintes planilhas:

**Obrigatórias:**
- `ADMISSÃO [MÊS].xlsx` - Funcionários admitidos no mês
- `ATIVOS.xlsx` - Lista de funcionários ativos
- `DESLIGADOS.xlsx` - Funcionários desligados
- `FÉRIAS.xlsx` - Funcionários em férias
- `Base dias uteis.xlsx` - Dias úteis por sindicato
- `Base sindicato x valor.xlsx` - Valores por estado

**Opcionais:**
- `AFASTAMENTOS.xlsx` - Funcionários afastados
- `EXTERIOR.xlsx` - Funcionários no exterior
- `APRENDIZ.xlsx` - Aprendizes
- `ESTAGIO.xlsx` - Estagiários

### 2. Execução

```bash
streamlit run app.py
```

### 3. Interface Web

1. **Configure os parâmetros:**
   - **Competência**: Mês/ano (ex: 2025-05)
   - **% Empresa**: Fração paga pela empresa (0.0-1.0)
   - **Ativar LLM**: Habilita supervisão inteligente

2. **Selecione o provedor LLM** (se ativado):
   - Gemini (recomendado)
   - OpenAI
   - OpenRouter

3. **Faça upload** do arquivo ZIP ou planilhas individuais

4. **Aguarde o processamento** e baixe os resultados

## 📊 Saídas do Sistema

### 1. Base Final
Planilha com os cálculos finais contendo:
- Matrícula do funcionário
- Data de admissão
- Sindicato
- Competência
- Dias trabalhados
- Valor diário VR
- Total
- Custo empresa
- Desconto profissional

### 2. Validações
Relatório de validações automáticas:
- Contagem de funcionários por categoria
- Verificações de integridade
- Alertas de inconsistências

### 3. Relatório Executivo
Relatório gerado por IA com:
- Resumo dos dados processados
- Explicações de validações
- Observações importantes

## 🔧 Configuração Avançada

### Provedores LLM Suportados

| Provedor | Modelos Padrão | Configuração |
|----------|----------------|--------------|
| **Gemini** | gemini-1.5-flash, gemini-1.5-pro | `GEMINI_API_KEY` |
| **OpenAI** | gpt-4o-mini, gpt-4o | `OPENAI_API_KEY` |
| **OpenRouter** | deepseek/deepseek-r1-0528 | `OPENROUTER_API_KEY` |

### Variáveis de Ambiente

```bash
# Chaves de API (opcional - podem ser inseridas na interface)
GEMINI_API_KEY=sua_chave_aqui
OPENAI_API_KEY=sua_chave_aqui
OPENROUTER_API_KEY=sua_chave_aqui
```

## 🧮 Regras de Negócio

O sistema aplica as seguintes regras:

### Exclusões Automáticas
- Estagiários
- Aprendizes
- Funcionários no exterior
- Afastados/Licenças
- Diretores

### Cálculos Proporcionais
- **Admissões**: Proporcional ao dia de admissão
- **Desligamentos**: 
  - Até dia 15: Exclusão total (se comunicado)
  - Após dia 15: Proporcional ao dia de desligamento

### Valores por Estado
- São Paulo: Padrão
- Rio Grande do Sul: Configurável
- Rio de Janeiro: Configurável
- Paraná: Configurável

## 🛠️ Desenvolvimento

### Estrutura do Código

```
projeto-i2a2/
├── app.py              # Interface Streamlit
├── agentes.py          # Orquestração de agentes
├── vrva_funcoes.py     # Lógica de negócio
├── requirements.txt    # Dependências
└── README.md          # Este arquivo
```

### Executando Testes

```bash
# Instalar dependências de desenvolvimento
pip install -r requirements.txt

# Executar aplicação
streamlit run app.py
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença [MIT](LICENSE).

## 🆘 Suporte

Para dúvidas ou problemas:

1. Verifique a seção de [Issues](../../issues)
2. Crie uma nova issue com detalhes do problema
3. Inclua logs de erro e exemplos de dados (sem informações sensíveis)

## 🔄 Changelog

### v1.0.0
- ✅ Sistema inicial de automação
- ✅ Interface Streamlit
- ✅ Integração com LLMs
- ✅ Processamento de planilhas
- ✅ Geração de relatórios

---

**Desenvolvido com ❤️ para automatizar processos de RH**
