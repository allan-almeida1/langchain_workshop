from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# 1. Definindo a estrutura da resposta desejada
class AgentResponse(BaseModel):
    analysis: str = Field(description="A brief analysis of the question")
    keypoints: List[str] = Field(description="List of keypoints of the answer")
    confidence: float = Field(description="Confidence level (0 ~ 1)")


# 2. Configurando o Parser e o Prompt
parser = PydanticOutputParser(pydantic_object=AgentResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a technical assistant. Always answer in the expected JSON format.\n{format_instructions}",
        ),
        ("user", "{question}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 3. Inicializando a LLM (ex: GPT-4o ou Claude)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 4. Construindo a Chain (LCEL - LangChain Expression Language)
chain = prompt | model | parser

# 5. Execução
user_question = "What are the advantages of using LangChain?"
result: AgentResponse = chain.invoke({"question": user_question})

print(f"Análise: {result.analysis}")
print(f"Confiança: {result.confidence}")
for keypoint in result.keypoints:
    print(f"- {keypoint}")
