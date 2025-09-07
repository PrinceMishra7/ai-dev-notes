import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from operator import add as add_messages
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

llm = ChatOllama(model="llama3.1",temperature=0)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

pdf_path = "Stock_Market_Performance_2024.pdf"


if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, # Overlap to maintain context
)

pages_split = text_splitter.split_documents(pages)

persist_directory = '/Users/prmishra2/Desktop/langgraph/db'
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # creating chroma db using embedding model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Created ChromaDB Vector Store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Create a retriever from the vectorstore

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query:str)->str:
    """
    This tool searches and returns the information from the Stock Market  Performance 2024 document.
    """

    docs = retriever.invoke(query)
    print(f"Retriever found {len(docs)} relevant documents.")

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []
    for i,doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]

llm.bind_tools(tools)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]


def should_continue(state:AgentState)->AgentState:
    """Check if last message contain tool calls"""
    result = state["messages"][-1]
    return hasattr(result,"tool_calls") and len(result.tool_calls)>0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


tools_dict = {tool.name:tool for tool in tools}

#LLM agent 
def call_llm(state:AgentState)->AgentState:
    """Function to call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)]+list(state["messages"])
    response = llm.invoke(messages)
    return {"messages":[response]}

# Retriever Agent 
def take_action(state:AgentState)->AgentState:
    """Executes the tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []

    for tool in tool_calls:
        print(f"Calling tool: {tool['name']} with query: {tool['arguments'].get('query','No Query Provided')}")

        if not tool['name'] in tools_dict:
            print(f"\nTool {tool['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[tool['name']].invoke(tool['args'.get('query', '')])
            print(f"Result length: {len(str(result))}")

        #Append the tool message

        results.append(ToolMessage(tool_call_id=tool['id'], name=tool['name'], content=str(result)))

    print("Tools Execution Completed. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm",call_llm)
graph.add_node("retriever",take_action)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True:"retriever",
        False:END
    }
)

graph.add_edge("retriever","llm")

app = graph.compile()

def running_agent():
    print("\n=== RAG AGENT ===")

    while True:
        user_input = input("\n👤 User: ")
        if user_input.lower() in ["exit","quit"]:
            break

        state = {"messages":[HumanMessage(content=user_input)]}

        result = app.invoke(state)

        print("=== FINAL RESPONSE ===")
        print(f"\n🤖 AI: {result['messages'][-1].content}")

running_agent()



