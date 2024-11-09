
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Load API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI Chat Model
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4")

# Sample existing rules (replace these with your actual rules)
existing_rules = [
    "Rule 1: All patients must provide a valid government-issued photo ID at check-in.",
    "Rule 2: Insurance information must be verified prior to admission.",
    "Rule 3: Patients without insurance must be informed about payment plans.",
    "Rule 4: Co-payments are collected at the time of service.",
    "Rule 5: Consent forms must be signed before any procedures.",
    "Rule 6: Emergency contact information must be updated annually.",
    "Rule 7: Patients under 18 must be accompanied by a parent or guardian.",
    "Rule 8: Allergies must be documented in the patient record.",
    "Rule 9: Advance directives should be requested during admission.",
    "Rule 10: Pre-authorization is required for elective surgeries."
]

# Create embeddings for the existing rules
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_texts(existing_rules, embeddings)

# Streamlit application
st.title("Rule Conflict Checker")

st.write("Enter a new rule to check for conflicts with existing rules.")

# User inputs the new rule
new_rule = st.text_area("New Rule", height=100)

if st.button("Check for Conflicts") and new_rule.strip():
    # Retrieve the most similar existing rules
    retrieved_docs = vectorstore.similarity_search(new_rule, k=5)
    
    # Prepare the context for the LLM
    context_rules = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create a prompt to ask ChatGPT to check for conflicts
    prompt_template = """
You are an expert in hospital administration and policy compliance.

Existing Rules:
{existing_rules}

New Rule:
{new_rule}

Analyze the new rule against the existing rules and identify any conflicts or overlaps. Provide a detailed explanation of any conflicts found. If there are no conflicts, confirm that the new rule does not conflict with existing rules.
"""
    prompt = prompt_template.format(existing_rules=context_rules, new_rule=new_rule)
    
    # Use the LLM to generate a response
    response = llm([HumanMessage(content=prompt)])
    
    # Display the response
    st.write("## Analysis")
    st.write(response.content.strip())
