#Reasoning and Acting Agent

#how to create a Tools in LangGraph
#different type of messages such as ToolMessage, SystemMessage,BaseMesage


from typing import Annotated,Sequence, TypedDict 
from langchain_core.messages import BaseMessage # the foundational class for all message types in langgraph
from langchain_core.messages import ToolMessage  # Passes data back to LLM after it calls a tool such as the content and tool_call_id 
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


# Annotated - provides additional context without affecting the type itself
# email = Annotated[str,"This has to be a valid email format!"]
# print(email.__metadata__)

# Sequence - To automatically handle the state updates for sequence such as by adding new message to a chat history

# add_messages - Reducer function
# Rule that controls how updates form nodes are combined with the existing state
# Tells us how to merge new data into the current state

# Without a reducer, updates woould have replaced the existing value entirely!
# in prev tutorials we were manually appending messages to state, but same can be done by add_messages now

#Without a reducer
# state = {"messages":["Hi!"]}
# update = {"messages":["How are you?"]}
# new_state = {"messages":["How are you?"]} #replaced

# #With a reducer
# state = {"messages":["Hi!"]}
# update = {"messages":["How are you?"]}
# new_state = {"messages":["Hi!","How are you?"]} #appended


console = Console()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]


@tool
def add(a:int,b:int)->int:
    """Adds two numbers"""
    return a+b

@tool
def multiply(a:int,b:int)->int:
    """Multiplies two numbers"""
    return a*b

@tool
def subtract(a:int,b:int)->int:
    """Subtracts second number from first """
    return a-b

tools = [add,multiply,subtract]

# Bind tools to the model
model = ChatOllama(model="llama3.1").bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability")
    # response = model.invoke(["You are my AI assistant, please answer my query to the best of your ability"])
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState)->AgentState:
    last_message = state["messages"][-1]
    print("Last message:",last_message, type(last_message))
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)

graph.add_node("model_call",model_call)
graph.set_entry_point("model_call")
graph.add_node("tool_call",ToolNode(tools=tools))

graph.add_edge("tool_call","model_call")

graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue":"tool_call",
        "end":END
    }
)

bot = graph.compile()


def print_stream(stream):
    # print("Streamed response: ",stream)
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user", "Add 40 and 12 and then multiply the result by 3 and also tell me a joke")]}
print_stream(bot.stream(inputs,stream_mode="values"))



'''Last message: content='' additional_kwargs={}
      response_metadata={
      'model': 'llama3.1', 
      'created_at': '2025-09-06T20:15:53.2577Z', 
      'done': True, 
      'done_reason': 'stop', 
      'total_duration': 10091690625, 
      'load_duration': 716469459, 
      'prompt_eval_count': 184, 
      'prompt_eval_duration': 8557805208, 
      'eval_count': 22, 
      'eval_duration': 816244917, 
      'model_name': 'llama3.1'
      } 
      id='run--ab41b406-4024-4141-93a3-15c3317a1987-0'
     tool_calls=[
     {'name': 'add', 'args': {'a': 4, 'b': 7}, 'id': '666c40ff-7a88-4894-9b79-dd447170f120', 'type': 'tool_call'} # this indicates that this tool call is suppose to be made
     ] 
     usage_metadata={'input_tokens': 184, 'output_tokens': 22, 'total_tokens': 206}
     '''





    
    









