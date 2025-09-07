from typing import TypedDict, List,Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="llama3.1")

def process(state:AgentState)->AgentState:
    """Calls the LLM"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    # console.print(Panel.fit(response.content.strip(), title="🤖 AI", subtitle="Llama3.1", style="cyan"))
    console.print("[bold cyan]🤖 AI[/bold cyan]:")
    console.print(Markdown(response.content.strip()))
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

bot = graph.compile()

conversation_history = []

user_input = console.input("[bold blue]👤 User[/bold blue]: ")
while user_input!="exit":
    # print("conversation_history:",conversation_history)
    conversation_history.append(HumanMessage(content=user_input))
    result = bot.invoke({"messages":conversation_history})
    conversation_history = result["messages"]
    user_input = console.input("[bold blue]👤 User[/bold blue]: ")

