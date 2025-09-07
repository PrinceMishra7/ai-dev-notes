from typing import TypedDict, List
from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv 
# load_dotenv()  

class AgentState(TypedDict):
    message:List[HumanMessage]

# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(model="llama3.1")

def process(state:AgentState)->AgentState:
    """"Calls the LLM with the current message"""
    response = llm.invoke(state['message'])
    print("Bot:", response.content)
    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

agent = graph.compile()
user_input = input("User: ")

while user_input != "exit":
    agent.invoke({"message":[HumanMessage(content=user_input)]})
    user_input = input("User: ")
