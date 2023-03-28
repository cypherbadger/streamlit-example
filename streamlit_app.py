import os

import streamlit as st
from llama_index import QuestionAnswerPrompt, GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
from langchain import OpenAI

# NOTE: for local testing only, do NOT deploy with your key hardcoded
# to use this for yourself, create a file called .streamlit/secrets.toml with your api key
# Learn more about Streamlit on the docs: https://docs.streamlit.io/
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]


index_name = "./index.json"
documents_folder = "./documents"

# load documents
documents = SimpleDirectoryReader(documents_folder).load_data()


# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_output))

# define custom QuestionAnswerPrompt
QA_PROMPT_TMPL = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

# Build GPTSimpleVectorIndex
index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)


@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    query_str = query_text
    response = _index.query(query_str, text_qa_template=QA_PROMPT)
    return str(response)


st.title("😎🏦 BH chatbot 🏦😎")
st.header("Bienvenido al demo de autogestion para atencion al cliente")
st.text("Ingrese su consulta sobre productos y servivios BH")

text = st.text_input("Query text:")

if st.button("Run Query") and text is not None:
    response = query_index(index, text)
    st.markdown(response)
