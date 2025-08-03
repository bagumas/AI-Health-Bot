import os
import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.tools import tool

# Set OpenAI API Key from Hugging Face Secrets
Open_API_Key = os.getenv("Open_API_Key")
os.environ["OPENAI_API_KEY"] = Open_API_Key

# Check if the key is loaded
if not os.getenv("Open_API_Key"):
    raise ValueError("OpenAI API key not found. Please set 'OPENAI_API_KEY' in your Hugging Face Space secrets.")

# Load FAISS Vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local('./disease_faiss_store', embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Define tools
@tool
def symptom_checker_tool(symptom_query: str) -> str:
    """Search for diseases based on described symptoms."""
    results = vectorstore.similarity_search(symptom_query, k=3)
    if not results:
        return "No matching diseases found."
    return "\n\n".join([doc.page_content for doc in results])

@tool
def precaution_advisor(symptom_query: str) -> str:
    """Advise on precautions based on described symptoms. Use this tool when a patient asks what they should do."""
    results = retriever.get_relevant_documents(symptom_query)
    if not results:
        return "No diseases or precautions found."
    response = []
    for doc in results:
        metadata = doc.metadata
        precautions = metadata.get('precautions', [])
        if precautions:
            response.append(f"- {metadata.get('disease')}:\n" + "\n".join([f"  • {p}" for p in precautions]))
    return "\n\n".join(response) if response else "No precautions available."

@tool
def severity_risk_estimator(symptom_list: str) -> str:
    """Estimate severity risk based on a list of symptoms."""
    symptoms = [s.strip() for s in symptom_list.split(',')]
    total_severity = 0
    known_symptoms = 0
    for s in symptoms:
        for doc in retriever.get_relevant_documents(s):
            for sym in doc.metadata.get('symptoms', []):
                if s.lower() in sym.lower():
                    known_symptoms += 1
                    total_severity += int(doc.metadata.get('severity', 1))
    if known_symptoms == 0:
        return "No known symptoms found."
    avg_sev = total_severity / known_symptoms
    risk = "Low" if avg_sev < 2 else "Medium" if avg_sev < 4 else "High"
    return f"Estimated risk: **{risk}** (Average Severity: {avg_sev:.2f})"

@tool
def symptom_to_disease_matcher(symptoms_query: str) -> str:
    """Find diseases matching multiple symptoms."""
    results = retriever.get_relevant_documents(symptoms_query)
    if not results:
        return "No diseases matched."
    return "\n".join([f"- {doc.metadata.get('disease')}: {doc.metadata.get('description', '')}" for doc in results])

@tool
def faq_disease_info(disease_name: str) -> str:
    """Get detailed information about a specific disease."""
    results = retriever.get_relevant_documents(disease_name)
    if not results:
        return f"No information for {disease_name}."
    doc = results[0]
    return (f"Disease: {disease_name}\n\nDescription: {doc.metadata.get('description', '')}\n\n"
            f"Symptoms: {', '.join(doc.metadata.get('symptoms', []))}\n\n"
            f"Precautions: {', '.join(doc.metadata.get('precautions', []))}")

tools = [symptom_checker_tool, precaution_advisor, severity_risk_estimator, symptom_to_disease_matcher, faq_disease_info]

# Setup OpenAI LLM & Memory
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)

# Define Agent Prompt
custom_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful health assistant. You MUST use the provided tools for health or symptom queries. "
     "If a tool returns no info, respond with: 'I cannot find information on that topic using my available tools.'"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

# Gradio Interface
def chat_with_agent(user_input, chat_history):
    result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    chat_history.append((user_input, result['output']))
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Health Assistant Chatbot")
    chatbot = gr.Chatbot()
    state = gr.State([])  # To hold chat history
    txt = gr.Textbox(placeholder="Ask about symptoms...", show_label=False)
    btn = gr.Button("Clear Chat")
    txt.submit(chat_with_agent, [txt, state], [txt, chatbot])
    btn.click(lambda: ([], []), None, [state, chatbot])

demo.launch(share=True)

