import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
# from langchain.schema.runnable import RunnableSequence
import json


st.set_page_config(page_title="💬 LLaMA 3 Chatbot with LangChain Memory")

llm = OllamaLLM(model="llama3.1")



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "memory" not in st.session_state:
    st.session_state.memory = memory



# prompt = PromptTemplate(
#     input_variables=["user_input"],
#     template= """
#                 You are an assistant that helps clean up coding prompts.
#                 For the given user input, do the following:
#                 1. Identify the programming language mentioned.
#                 2. Rephrase the prompt to make it clear and professional.

#                 If the language or prompt cannot be identified, return "NA" for the value.

#                 ⚠️ Important:
#                 - Respond with ONLY a JSON object.
#                 - DO NOT add any explanation, markdown, or extra text.
#                 - Format must be exactly:
#                 {{
#                 "language": "<detected language or 'NA'>",
#                 "rewritten_prompt": "<rewritten version or 'NA'>"
#                 }}

#                 Now process this input:
#                 "{user_input}"
#             """
# )


prompt = PromptTemplate(
    input_variables=["chat_history","user_input"],
    template= """
        You are a helpful and intelligent assistant that communicates clearly and professionally.

        Your job is to:
        - Understand user questions, even if they're casual or vague.
        - Respond in a friendly and concise way.
        - Use bullet points or code blocks if needed to improve clarity.
        - If the user asks for code, generate clean, well-commented code in the requested language.
        - If you're unsure about something, say so honestly instead of guessing.

        Always keep your tone helpful, neutral, and to the point.

        You are chatting with a user. Here is the conversation so far:
        {chat_history}

        User question:
        {user_input}
    """
)


chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=st.session_state.memory,
    verbose=False,  
)

# user_input = input("Enter your prompt : ")
st.title("💬 LLaMA 3 Chatbot with Memory")
st.markdown("Ask anything! The assistant remembers your past questions.")

user_input = st.text_input("User : ")

if user_input:
    with st.spinner("Thinking..."):
        response = chain.run(user_input) # chat_history goes by itself no need to pass it
    
    for msg in reversed(st.session_state.memory.chat_memory.messages):
        role = "User" if msg.type == "human" else "Assistant"
        st.markdown(f"**{role}:** {msg.content}")


# chain = prompt | llm# chain = RunnableSequence(

# response = chain.invoke({"user_input":user_input})

# print(f"Raw response from LLM: {response}")

# try:
#     response = json.loads(response)
# except json.JSONDecodeError:
#     print("Error decoding JSON response")
#     response = {"language": "NA", "rewritten_prompt": "NA"}

# print(f"Response from LLM: {response.get("language")} {response.get("rewritten_prompt")}")

