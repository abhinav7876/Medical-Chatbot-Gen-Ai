from flask import Flask, render_template, request
from src.helper import download_embeddings,hyde_query_expansion,rerank_docs
from evaluation.eval import evaluate_with_threshold
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from sentence_transformers import CrossEncoder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv
from src.prompt import *
from flask import session
import uuid
import os
import time

load_dotenv()
os.environ["PINECONE_API_KEY"] =os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= "true"
app = Flask(__name__)
app.secret_key = "super-secret-key"
chatModel = ChatOpenAI(model="gpt-4o")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOpenAI(model="gpt-4o-mini")



embedding = download_embeddings()
index_name="medical-chatbot-app"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":8})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="history"),
    ]
)
store={}
def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in store:
            store[session_id]=ChatMessageHistory()
        return store[session_id]

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    start_time=time.time()
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4()) 
    session_id = session["session_id"]
    session_history = get_session_history(session_id)
    msg = request.form["msg"]
    input = msg
    print(input)
    contextualize_q_system_prompt = """
            You are a query rewriter.
            Rewrite follow-up questions into standalone questions using chat history.
            Resolve pronouns like 'it', 'they', 'this'.
            Do not answer the question, only rewrite it if needed.
            """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    standalone_question = llm.invoke(contextualize_q_prompt.format(input=input,history=session_history.messages)).content
    std_q_time=time.time()
    print("Standalone question time:", std_q_time - start_time)
    pseudo_answer=hyde_query_expansion(standalone_question,llm)
    pseudo_answer_time=time.time()
    print("Pseudo answer time:", pseudo_answer_time - std_q_time)
    retrieved_docs = retriever.invoke(pseudo_answer)
    retrieved_docs_time=time.time()
    print("Retrieved docs time:", retrieved_docs_time - pseudo_answer_time)
    top_docs = rerank_docs(standalone_question, retrieved_docs,reranker)
    reranked_docs_time=time.time()
    print("Reranked docs time:", reranked_docs_time - retrieved_docs_time)
    final_context="\n\n".join([d.page_content for d in top_docs])
    rag_chain = (
        RunnableParallel(
            {
                "context": lambda _: final_context,
                "input": lambda _: standalone_question,
                "history": lambda x: x["history"],
                
            }
        )
        | prompt
        | chatModel
    )

    rag_chain_with_memory = RunnableWithMessageHistory(rag_chain,get_session_history,
                 input_messages_key="input",
                 history_messages_key="history")
    response = rag_chain_with_memory.invoke(
                {"input": standalone_question},
                    config={"configurable": {"session_id": session_id}})
    response_time=time.time()
    print("Response generated time:", response_time - reranked_docs_time)                 
    final_response,scores=evaluate_with_threshold(standalone_question,final_context,response.content,chatModel,session_id)
    print("Final Response generated time:", time.time() - float(reranked_docs_time))   
    #print("Final Response : ", response.content)
    print("scores are: ",scores)
    #return str(response.content)
    print("Evaluation time:", time.time() - response_time)
    print("Total time taken:",time.time() - start_time)
    return str(final_response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080,debug=True)