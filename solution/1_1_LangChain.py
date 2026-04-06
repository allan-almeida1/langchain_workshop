from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# 1. Definindo a estrutura da resposta desejada
class RespostaAgente(BaseModel):
    analise: str = Field(description="Uma análise breve da pergunta")
    pontos_chave: List[str] = Field(
        description="Lista de pontos principais da resposta"
    )
    confianca: float = Field(description="Nível de confiança de 0 a 1")


# 2. Configurando o Parser e o Prompt
parser = PydanticOutputParser(pydantic_object=RespostaAgente)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um assistente técnico. Responda sempre no formato JSON esperado.\n{format_instructions}",
        ),
        ("user", "{pergunta}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 3. Inicializando a LLM (ex: GPT-4o ou Claude)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 4. Construindo a Chain (LCEL - LangChain Expression Language)
chain = prompt | model | parser

# 5. Execução
pergunta_usuario = "Quais as vantagens de usar o LangChain?"
resultado: RespostaAgente = chain.invoke({"pergunta": pergunta_usuario})

print(f"Análise: {resultado.analise}")
print(f"Confiança: {resultado.confianca}")
for ponto in resultado.pontos_chave:
    print(f"- {ponto}")
