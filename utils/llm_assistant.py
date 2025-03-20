"""
llm_assistant.py

this file contains the implementation of the Assistant and State Class
An Assistant Class backed by connection to an LLM 
    ==> LLM is either an active connection to openAI or ChatGroq, If OPenAI fails

State Class: 
    ==> This is the simple implememtation of how messages from user to llm would be tracked and stored

"""

from langchain_core.runnables import Runnable
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    A dictionary-like structure for maintaining conversation state.

    This class defines the structure of the state, which stores messages exchanged 
    between the user and the assistant.

    Attributes:
    -----------
    messages : list
        A list of messages that represent the conversation history. The `add_messages` 
        function determines how new messages are appended instead of being overwritten.
    """

    # Messages have the type "list". decorated with The `add_messages` function
    #the add_messages appends messages to the list, rather than overwriting them
    messages: Annotated[list, add_messages]


class Assistant:
    """
    A conversational assistant that interacts with a runnable object to process user queries.

    The assistant continuously prompts the runnable object until a valid response is obtained.

    Methods:
    --------
    __init__(runnable: Runnable):
        Initializes the assistant with a given runnable object.

    __call__(state: State) -> dict:
        Processes the conversation state, ensuring a valid response is generated.

        Parameters:
        -----------
        state : State
            A dictionary containing the conversation history.

        Returns:
        --------
        dict
            A dictionary containing the updated conversation messages.
    """

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State):
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
