import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS

MODEL_DIR = '/Users/sakthi/Documents/PyProjects/opensource_pdf_search/models/LaMini-Flan-T5-783M'
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    # device_map='cpu',
    torch_dtype=torch.float32
)


# @st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        do_sample=True,
        temperature=0.3,
        top_p=0.90
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


# @st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_TRANSFORMER_MODEL)
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def process_answer(instruction):
    response = ''
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text


def main():

    st.title('Search Your PDF')
    with st.expander("About the app"):
        st.markdown(
            '''
            This is a Generative AI powered Question-Answering app that responds to the questions about your PDF files.
            '''
        )
    question = st.text_area("Enter your question: ")
    if st.button("Search"):
        st.info(f"Your Question: {question}")
        st.info("Your Answer: ")
        answer, metadata = process_answer(instruction=question)
        st.write(answer)
        st.write(metadata)


if __name__ == "__main__":
    main()
