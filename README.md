# LangChain & LangGraph Workshop

## Configuração e Dependências

1. Instalar o UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Criar ambiente virtual
```bash
uv init
```

3. Instalar dependências
```bash
uv add langchain langchain-openai langgraph langchain-tavily python-dotenv
```

## Conceitos

Os conceitos básicos para a compreensão do LangChain e LangGraph, assim como o planejamento deste workshop, estão descritos no [documento de apresentação](./docs/workshop.pdf)

## LangChain Hands-On

**Desafio 1**: Criar uma chain que receba uma pergunta, gere uma resposta através da LLM e faça o parsing da resposta.

**Desafio 2**: Criar um agente ReAct de busca na web usando LangChain.

## LangGraph Hands-On

**Desafio 3**: Criar um assistente de compras que recebe uma lista manuscrita de itens e gera uma estimativa de gasto total.
