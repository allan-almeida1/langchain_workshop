import base64
from operator import add
from pathlib import Path
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()


# 1. Definindo as classes base
class Product(BaseModel):
    name: str = Field(description="Name of the product")
    price: float = Field(description="Price of the product")
    description: Optional[str] = Field(
        default="", description="Description of the product"
    )


class CartItem(BaseModel):
    product: Product = Field(description="Details of the product")
    amount: int = Field(description="Amount of this item in the cart")

    @property
    def subtotal(self) -> float:
        return self.product.price * self.amount


class Cart(BaseModel):
    items: Annotated[List[CartItem], add] = Field(
        description="List of items and their amount in the cart"
    )

    @property
    def total_price(self) -> float:
        return sum(item.subtotal for item in self.items)


# 2. Definindo o estado
class State(BaseModel):
    cart: Cart = Field(description="Cart of items")
    image_path: Path = Field(
        description="Path of the image file containing the handwritten shopping list"
    )
    products_not_found: List[str] = Field(
        description="List containing the names of the products not found by the agent"
    )
    shopping_list: dict = Field(
        description="Dictionary containing item name and amount"
    )
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        description="List of messages exchanged during the workflow"
    )


# 3. Definindo as tools
@tool
def local_store_search(product_name: str) -> Product | None:
    """Search for products on the local store, which has the best products
    and the best prices.

    Args:
        product_name (str): The name of the product

    Returns:
        Product | None: The product if found, otherwise None
    """
    inventory = {
        "beans": Product(name="Beans", price=8.99, description="Package of beans"),
        "rice": Product(name="Rice", price=6.99, description="Package of white rice"),
        "black_pepper": Product(
            name="Black Pepper", price=4.90, description="Package of black pepper"
        ),
        "salt": Product(name="Salt", price=3.49, description="Package of salt"),
        "wheat_flour": Product(
            name="Wheat Flour", price=5.99, description="Package of wheat flour"
        ),
        "toilet_paper": Product(
            name="Toilet Paper",
            price=12.99,
            description="Package containing 8 rolls of toiled paper",
        ),
        "tomato": Product(name="Tomato", price=0.50, description="Unit of tomato"),
    }
    product = inventory.get(product_name, None)
    return product


tools = [local_store_search]


# 4. Definindo os nodes
def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def list_extractor(state: State):
    # Codificar a imagem que está no estado
    base64_image = encode_image(state.image_path)

    # Instanciar a LLM
    llm = ChatOpenAI(model="gpt-4.1-mini")

    # Monta a mensagem
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """Extract the items and amounts from this handwritten
                list. Return only a JSON containing each product name and the
                corresponding amount. 
                
                IMPORTANT RULES:
                1. If the name is composed of more than one word, write the 
                name separated by underscores
                2. If the name of the item makes sense in singular form 
                (e.g. orange) write it in singular, otherwise (e.g. beans) write
                it in plural
                
                Example of a correct response:
                {
                    \"beans\": 3,
                    \"rice\": 4,
                    \"wheat_flour\": 1         
                }""",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )

    parser = JsonOutputParser()

    chain = llm | parser

    response: dict = chain.invoke([message])

    return {
        "shopping_list": response,
        "messages": [AIMessage(content=str(response))],
    }


def shopper(state: State):
    # Instanciar a LLM
    llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

    # Definir parser
    cart_parser = PydanticOutputParser(pydantic_object=Cart)

    # Definir prompt
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant which receives a shopping list and builds
                a shopping cart for the user.
                
                Here is the shopping list: {shopping_list}
                
                You have access to tools. For each item:
                
                1. Search in the local store
                2. If the item is null, ignore it and move to the next
                
                Format found items as: {cart}
                """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt_template | llm

    response = chain.invoke(
        {
            "shopping_list": state.shopping_list,
            "cart": cart_parser.get_format_instructions(),
            "messages": state.messages,
        }
    )

    if response.tool_calls:
        return {"messages": [response]}

    try:
        parsed_cart = cart_parser.invoke(response)
        return {"messages": [response], "cart": parsed_cart}
    except Exception as e:
        print(f"Error parsing cart: {e}")


def missing_items(state: State):
    shopping_list = [key for key in state.shopping_list.keys()]
    cart_items = [item.product.name for item in state.cart.items]

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    parser = JsonOutputParser()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                From all items given in the shopping list: {shopping_list}
                
                Identify the items that are MISSING in the cart: {cart_items}
                and return JSON containing the list of missing items.
                
                IMPORTANT: 
                1. Return only the list containing missing items, nothing else, no extra text
                2. Don't match names exactly, wheat_flour is the same as Wheat Flour
                """,
            )
        ]
    )

    chain = prompt_template | llm | parser

    response = chain.invoke({"shopping_list": shopping_list, "cart_items": cart_items})
    return {"products_not_found": response}


tool_node = ToolNode(tools)


LIST_EXTRACTOR = "list_extractor"
SHOPPER = "shopper"
TOOLS = "tools"
PRODUCTS_NOT_FOUND = "products_not_found"


def should_continue(state: State):
    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if tool_calls:
        return "tools"

    return PRODUCTS_NOT_FOUND


# 5. Construindo o grafo
def build_graph():
    graph = StateGraph(State)

    graph.add_node(LIST_EXTRACTOR, list_extractor)
    graph.add_node(SHOPPER, shopper)
    graph.add_node(TOOLS, tool_node)
    graph.add_node(PRODUCTS_NOT_FOUND, missing_items)

    graph.add_edge(START, LIST_EXTRACTOR)
    graph.add_edge(LIST_EXTRACTOR, SHOPPER)
    graph.add_conditional_edges(
        SHOPPER,
        should_continue,
        {"tools": TOOLS, PRODUCTS_NOT_FOUND: PRODUCTS_NOT_FOUND},
    )
    graph.add_edge(TOOLS, SHOPPER)
    graph.add_edge(PRODUCTS_NOT_FOUND, END)
    return graph.compile()


# 6. Executando o grafo
agent = build_graph()

initial_state = State(
    cart=Cart(items=[]),
    image_path=Path("shopping_list.jpeg"),
    products_not_found=[],
    shopping_list={},
    messages=[],
)

result = agent.invoke(initial_state)

print("Cart:")
cart: Cart = result["cart"]
for item in cart.items:
    print(
        f" - {item.product.name}: {item.amount} - ${item.product.price * item.amount:.2f}"
    )
print(
    "Total price: ${:.2f}".format(
        sum(item.product.price * item.amount for item in cart.items)
    )
)
print("Products Not Found:", result["products_not_found"])
