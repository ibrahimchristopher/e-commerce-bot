import re
import openai
from dotenv import load_dotenv


"""
create_agent.py

this function is the implementation and creation of the chatbot autonomous agent handler for user reqquest:

it has the following features/Implementation steps:

Creating A Vectorstore Object containing FAQ and Policy for Agent to consult
    ==> Vectorstore is as a result of embeddings created from OPen AI or the whole docs as fall back

An Agent that with the assistant at its core has the full abilty specified by the deliverables
    ==> this class has the most relevant method; query().
        this method recives user input and address thier questions or requests
        
        An Assistant object backed by connection to an LLM 
            ==> LLM is either an active connection to openAI or ChatGroq, If OPenAI fails


!!!!Implementing these objects one at a time in the order they were mentioned
"""


##first lets load up env variables
# Load environment variables from .env file
load_dotenv()

##setting up vector store doing the following: 
# read policy, 
# chunk it and filter out any issues, 
# lastly create a retriver

from vectorstore import VectorStoreRetriever

from agentic_system import AgenticSystem #Assistant object creation already handled by this import

def create_agent():

    # Read policies from a local file at ##policy.txt
    with open("policy.txt", "r", encoding="utf-8") as file:
        faq_text = file.read()

    #set up chunks
    docs = [{"page_content": txt.strip()} for txt in re.split(r"(?=Q:)", faq_text)]

    #filter out any problematic issues here
    docs = [doc for doc in docs if len(doc["page_content"]) > 0]

    # creating a retriever object or a default object if it doesnt work!!!
    try:
        retriever = VectorStoreRetriever.from_docs(docs, openai.Client())
    except:
        retriever = None

    agent = AgenticSystem(retriever=retriever)

    return agent