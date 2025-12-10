import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

def set_custom_prompt(template: str):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",  # or your preferred Groq model
        temperature=0.0,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=1
    )

def main():
    st.title("Ask Chatbot!")
    st.sidebar.info("Powered by xAI’s Groq via LangChain")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    user_input = st.chat_input("Pass your prompt here")
    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role':'user', 'content': user_input})

        prompt_template = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer.
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(prompt_template)}
            )

            response = qa_chain.invoke({'query': user_input})
            result = response["result"]
            sources = response["source_documents"]
            source_info = "\n\n**Source Documents:**\n" + "\n\n".join(
                f"- {doc.metadata.get('source','unknown')}: {doc.page_content[:200]}..."
                for doc in sources
            )
            assistant_msg = result + source_info
            st.chat_message('assistant').markdown(assistant_msg)
            st.session_state.messages.append({'role':'assistant', 'content': assistant_msg})
        
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
