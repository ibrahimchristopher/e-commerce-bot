import os
import pandas as pd
import uuid
import numpy as np
import sqlite3
import re
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict
import openai
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

#####To set up my agent, i first start with creating a vector store where policies can be fetched
#####this would be useful in setting up my Tools Node



# Load environment variables from .env file
load_dotenv()

# Read policies from a local file at ##policy.txt
with open("policy.txt", "r", encoding="utf-8") as file:
    faq_text = file.read()

#set up chunks
docs = [{"page_content": txt.strip()} for txt in re.split(r"(?=Q:)", faq_text)]

#filter out any problematic issues here
docs = [doc for doc in docs if len(doc["page_content"]) > 0]

class VectorStoreRetriever:
    """
    class implemetation for setting up and querying vector store
    """
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        # Extract the text content for embedding
        input_texts = [doc["page_content"] for doc in docs]
        
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=input_texts
        )
        
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]



# creating a retriever object
retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


###########Next up setting my Assistant

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


##helper function to be used in the AgenticSystem Class
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


class AgenticSystem:
    def __init__(self, ):
        #setting up my llm
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.db_file  = "orders.db"###filepath to example database we set up
        #tools avalaible for the A.I. assistant to use
        self.dev_tools = [
            self.check_order_status,
            self.request_human_rep,
            self.lookup_policy,

        ]
        
        self.graph = self.build_graph()

    def check_order_status(self, order_id: str):
        """
        this function recives an order id and uses it to query a database for order status and order date
        """
        if not order_id:
            raise ValueError("No Order ID passed.")
        # Connect to the database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        # Query to check the order details based on order_id
        cursor.execute(
            "SELECT order_id, status, order_date FROM orders WHERE order_id = ?",
            (order_id,)
        )
        order = cursor.fetchone()

        if not order:
            cursor.close()
            conn.close()
            return f"Order with ID {order_id} not found."
        # Retrieve the order details
        order_id_val, status, order_date = order

        # Prepare the response with more details
        response = (
            f"Order ID: {order_id_val}\n"
            f"Status: {status}\n"
            f"Order Date: {order_date}\n"
        )

        # Close the database connection
        cursor.close()
        conn.close()

        return response


    def request_human_rep(self, full_name: str, email: str, phone_number: str) -> str:
        """Request human representative contact information and save to a CSV file using pandas.

        ALways check that the info has been provided at all times

        Args:
            full_name (str): The user's full name.
            email (str): The user's email address.
            phone_number (str): The user's phone number.
        
        Returns:
            str: Confirmation message with file path.
        """
        
        filename = f"{full_name.replace(' ', '_')}.csv"
        # Create a DataFrame with the contact information
        contact_info = {
            'Full Name': [full_name],
            'Email': [email],
            'Phone Number': [phone_number]
        }
        df = pd.DataFrame(contact_info)
        print(df)
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        
        # Return a confirmation message
        return f"Your contact information has been saved"
    
    def lookup_policy(self, query: str) -> str:
        """Consult the company policies.
        look up the vector store and summarize the returned value to end user by llm
        """
        docs = retriever.query(query, k=2)
        return "\n\n".join([doc["page_content"] for doc in docs])

    
    def get_assistant(self, ) -> Runnable:
        
        dev_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful customer support assistant for an e commerce site. "
                    " Assist the user with their enquiries, save details to csv files when they want to speak to a rep, look up policy  and run look up SQL queries where necessary. "
                    " After you are able to discern all the information, call the relevant tool."
                    " After calling a tool, ensure to summarize and present the output to the user in a suitable format."
                    "\nCurrent time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now)

        dev_runnable = dev_prompt | self.llm.bind_tools(self.dev_tools)
        return dev_runnable

    def build_graph(self,):
        # graph builder
        graph_builder = StateGraph(State)

        # nodes
        dev_runnable = self.get_assistant()
        graph_builder.add_node("assistant", Assistant(dev_runnable))
        graph_builder.add_node("tools", ToolNode(self.dev_tools))

        # edges
        graph_builder.add_edge(START, "assistant")
        graph_builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        graph_builder.add_edge("tools", "assistant")

        # compile graph
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        return graph

    def stream_graph_updates(self, user_input: str, config: dict, verbose: bool=True) -> str:
        _printed = set()
        for event in self.graph.stream({"messages": ("user", user_input)}, config, stream_mode="values"):
            for value in event.values():
                if verbose:
                    _print_event(event, _printed)
        return value

    def run(self, config: dict=None):
        if config is None:
            config = {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                }
            }
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.stream_graph_updates(user_input, config=config)
            except:
                break

    def query(self, query: str, config: dict=None, verbose: bool=False, last_message_only: bool=True):
        if config is None:
            config = {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                }
            }
        response = self.stream_graph_updates(user_input=query, config=config, verbose=verbose)
        if last_message_only:
            return response[-1].content
        return response

