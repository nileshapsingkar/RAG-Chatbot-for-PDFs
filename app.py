import streamlit as st 
import tempfile 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI

#-------------------------------------------------------------------------

#""" this function gets the pdf, extracts the content, splits it into chunks, stores the vector in FAISS vector store and returns the retriever """
def process_pdf(file, api_key):
    # creates a temp file of the uploaded pdf, and ensures it is not deleted
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        temp_file_path = tmp_file.name

    # load the pdf through pypdf
    loader = PyPDFLoader(temp_file_path)
    pdf_data = loader.load()

    # make processing the content of the pdf easier for analysis
    text_splitter = CharacterTextSplitter(
        separator="\n", # split based on new line
        chunk_size=1000, #1000 char
        chunk_overlap=150, # for correct context and link between chunks
        length_function=len # use the len of the text in terms of char to split
    )
    
    docs = text_splitter.split_documents(pdf_data)

    # initialize the embeddings through OpenAI, to generate the vector representation of the document
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # store the vector in FAISS as db
    db = FAISS.from_documents(docs, embedding=embeddings)

    # return a retriever so that it fetches top 3 documents
    return db.as_retriever(search_kwargs={'k': 3}) 


#--------------------------------------------------

#""" Main streamlit app """
def run_app():
    st.title("RAG Chat Bot for PDFs") 

    # set the session state
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    
    if "is_querying" not in st.session_state:
        st.session_state.is_querying = False

    # get the user inputs - api key and the pdf
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
    
    if openai_api_key and uploaded_pdf:
        if st.session_state.retriever is None:
            with st.spinner("Processing your PDF..."):
                # calling the function to process pdf and extract content
                st.session_state.retriever = process_pdf(uploaded_pdf, openai_api_key) 
                st.success("PDF processed successfully!")

        # set up the QA chain
        retriever = st.session_state.retriever
        # initialize the OpenAi model
        model = ChatOpenAI(api_key=openai_api_key)
        
        # provide the instructions to the model
        custom_prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Answer:
        """
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])

        # create a chain using the model, and the retriever from processing the pdf
        qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff", # fetch necessary docs, and use them to answer
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # take the question from the user, this will be disabled when a query is being processed 
        question = st.text_input("Ask a question about the document", disabled=st.session_state.is_querying)

        
        if question and not st.session_state.is_querying:
            # Check if the question is the same as the most recent question
            if len(st.session_state.qa_history) == 0 or st.session_state.qa_history[0]['question'] != question:
                st.session_state.is_querying = True
                with st.spinner("Retrieving the answer..."):
                    # Run the query
                    result = qa({"query": question})
                    answer = result['result']
                    source_documents = result['source_documents']

                    # to store the history of question and answer
                    # Prepend to history (newest first)
                    st.session_state.qa_history.insert(0, {"question": question, "answer": answer, "sources": source_documents})
                st.session_state.is_querying = False

        # Display question and answer history
        st.write("### Question & Answer History")
        for i, entry in enumerate(st.session_state.qa_history):
            st.write(f"**Q{i + 1}:** {entry['question']}")
            st.write(f"**A{i + 1}:** {entry['answer']}")

            # Toggle button to show sources for each question
            if st.button(f"Toggle Sources for Q{i + 1}"):
                with st.expander(f"Source Documents for Q{i + 1}"):
                    for doc in entry["sources"]:
                        # Accessing the page content correctly
                        st.write(doc.page_content)
        
if __name__ == "__main__":
    run_app()
