import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

# Runs locally on your machine — no API key needed
embeddings = HuggingFaceEmbeddings(model_name="./local_model")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_vectorstore(chunks):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[RAG] Saved {len(chunks)} chunks to FAISS index.")
    return vectorstore


def load_vectorstore():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def answer_question(question: str) -> str:
    vectorstore = load_vectorstore()

    if vectorstore is None:
        return "No document uploaded yet. Please upload a PDF or TXT file first."

    # Free Groq LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on the document."

Context: {context}

Question: {question}
""")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)