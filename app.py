import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set upi the Stramlit app
st.set_page_config(
    page_title="Text-to-Math Solver & Smart Data Assistant",
    page_icon="🧮"
)

# Catchy and friendly app title
st.title("🔍 Solve Math Problems with Text – Powered by Gemma 2")

# Sidebar input for API key with a helpful hint
groq_api_key = st.sidebar.text_input(
    label="🔑 Enter Your Groq API Key",
    type="password",
    help="Your key stays private and is only used for this session."
)

# Informative message if API key is not provided
if not groq_api_key:
    st.info("🚨 To get started, please enter your Groq API key in the sidebar.")
    st.stop()


llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"

)

## Initializa the MAth tool

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explaination
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

## initialize the agents

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a Math chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## Lets start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("🔍 Find My Answer"):
    if question:
        with st.spinner("🤖 Thinking... Generating your response..."):
            # Add user message to session
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Callback for displaying thought process
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            # Generate response using the assistant agent
            response = assistant_agent.run(
                st.session_state.messages,
                callbacks=[st_cb]
            )
            
            # Store and display the assistant's response
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write("### ✅ Here's what I found:")
            st.success(response)
    else:
        st.warning("⚠️ Please enter your question above before clicking the button.")