from flask import Flask, render_template, request
from src.helper import download_embeddings,hyde_query_expansion,rerank_docs
from evaluation.eval import evaluate_with_threshold
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableConfig
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from sentence_transformers import CrossEncoder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv
from uuid import uuid4
from src.prompt import *
import os
load_dotenv()
os.environ["PINECONE_API_KEY"] =os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

chatModel = ChatOpenAI(model="gpt-4o")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOpenAI(model="gpt-4o-mini")



embedding = download_embeddings()
index_name="medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":8})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{context}\n\n" + system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

store={}
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    pseudo_answer=hyde_query_expansion(input,llm)
    retrieved_docs = retriever.invoke(pseudo_answer)
    top_docs = rerank_docs(input, retrieved_docs,reranker)

    final_context="\n\n".join([d.page_content for d in top_docs])
 
    rag_chain = (
        RunnableParallel(
            {
                "context": lambda _: final_context,
                "input": RunnablePassthrough(),
                "chat_history": lambda x: x.get("chat_history", [])
            }
        )
        | prompt
        | chatModel
    )
    rag_chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    )
    from uuid import uuid4

    session_id = "user1"   
    response = rag_chain_with_memory.invoke(
                {"input": msg},
                    config={"configurable": {"session_id": session_id}},)
    print("session_id before",session_id)
    #response = rag_chain_with_memory.invoke({"input": msg})
    final_response,_=evaluate_with_threshold(msg,final_context,response.content)
    print("session_id after",session_id)
    print("Final Response : ", final_response)
    return str(final_response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)