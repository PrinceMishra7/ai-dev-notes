from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode



document_content = ""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]


@tool
def update(content:str)->str:
    """Updates the document with provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The Current content is: {document_content}"

@tool
def save(filename:str)->str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = filename+".txt"

    try:
        with open(filename,"w") as file:
            file.write(document_content)
        print(f"\nDocument has been saved successfully as {filename}")
        return f"Document has been saved successfully as '{filename}'."
    except Exception as e:
        return f"Error occurred while saving document: {str(e)}"
    

tools = [update,save]

model = ChatOllama(model="llama3.1").bind_tools(tools)


def drafter_agent(state:AgentState)->AgentState:
    system_prompt =SystemMessage(
        content=f"""
    You are a Drafter, a helpful writing assistant. You are going to help the user update and modify a documents.

    - if the user wants to update or modify the content, use the 'update' tool with the complete updated content.    
    - if the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.

    The current document content is: {document_content} 
    """
    )

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. what would you like to create?"
        user_message = HumanMessage(content="Hi! I am your drafter assistant. How can I help you?")
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 User: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🛠 USING TOOLS: {[tool_call['name'] for tool_call in response.tool_calls]}")
    
    return {"messages": list(state["messages"])+ [user_message, response]}

def should_continue(state:AgentState)->AgentState:
    """Determines if we should continue or end the conversation."""

    messages = state["messages"]
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if(isinstance(message, ToolMessage) and 
        "saved" in message.content.lower() and 
        "document" in message.content.lower()):
            return "end"

    return "continue"
              
def print_messages(messages):
    """prints the messages in a readable format"""

    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠 TOOL RESULT:- {message.content}")



graph = StateGraph(AgentState)
graph.add_node("drafter_agent",drafter_agent)
graph.set_entry_point("drafter_agent")

graph.add_node("tools",ToolNode(tools=tools))

graph.add_edge("drafter_agent","tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"drafter_agent",
        "end":END
    }

)

bot = graph.compile()

def run_draft_agent():
    print("\n ==== DRAFTER ====")

    state = {"messages":[]}

    for step in bot.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ==== DRAFTING ENDED ====")


if __name__ == "__main__":
    run_draft_agent()



