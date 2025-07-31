from langchain_groq import ChatGroq  
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
#Step 1: Setup LLm (Use deepseek)
#load_dotenv
##llm_model = ChatGroq("deepseek-r1-distill-llama-70b")
#Step 2: Retreive docs
from dotenv import load_dotenv
import os

load_dotenv()  # Load the .env file

llm_model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",  # or your specific model like "llama3-70b-8192"
    groq_api_key="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"
)

##def retreive_docs(query):
 ##   faiss_db.similarity_search(query)
def retrieve_docs(query):
    return faiss_db.similarity_search(query)  

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


#Step 3: Answer Question
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""
def answer_query(documents,model,query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})