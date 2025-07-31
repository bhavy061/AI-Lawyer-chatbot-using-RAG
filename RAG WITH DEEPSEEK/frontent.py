from rag_pipeline import answer_query, retrieve_docs,llm_model
#Step 1 Setup upload PDF functionality
import streamlit as st
uploaded_file = st.file_uploader("Upload PDF",type="pdf",
                                 accept_multiple_files=False)
#step 2 Chatbot skeleton (ques and ans)
user_query = st.text_area('Enter your prompt ',height=150,placeholder="Ask anything")
ask_question = st.button("Ask AI Lawyer")
if ask_question:
    if uploaded_file:
        st.chat_message("user").write(user_query)
    #RAG Pipeline
        retrived_docs =  retrieve_docs(user_query)
        answer = answer_query(documents=retrived_docs,model=llm_model,query=user_query)
       # fixed_response = "Hii this is the fixed response"
        st.chat_message("AI Lawyer").write(answer)
    else:
        st.error("Kindly upload a valid PDF file")