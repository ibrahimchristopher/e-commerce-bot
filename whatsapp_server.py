##Absoluttely optional part of the pipeline but considering we need an outlet for people to report issues, 
#why not a whatsapp channel
##All you need is a twilo account and free ngrok api

#see setup.txt on  how to run this

import os
from fastapi import FastAPI, Form, Depends

# Internal imports
from utils import send_message, logger
import uuid
from agent import AgenticSystem

#setting up agentic system and thread to save context
agent = AgenticSystem()
user_threads = {}


app = FastAPI()
MAX_LENGTH = 1600

@app.post("/message")
async def reply(Body: str = Form(), From: str = Form()):
    """
    Handles incoming messages from WhatsApp and sends a reply.

    Args:
        Body (str): The message body sent by the user.
        From (str): The WhatsApp number of the sender.
        db (Session): The database session dependency.
    """
    # Log the incoming message and sender
    print(f'recived message from {From}')
    logger.info(f"Received message from {From}: {Body}")

    #retrievig thread _id to cintinue xonveration or creating a new one
    thread_id = user_threads.get(From, str(uuid.uuid4()))
    user_threads[From] = thread_id

    # Call the OpenAI API to generate a response


    config = {
    "configurable": {
        "thread_id": thread_id,
            }
        }


    query = Body
    
    response = agent.query(query=query, config=config, last_message_only=False)
   
    chat_response = response[-1].content

    phone_number = From.split(':')[1]
    
    if len(chat_response) > MAX_LENGTH:
        chunks = [chat_response[i:i+MAX_LENGTH] for i in range(0, len(chat_response), MAX_LENGTH)]
        for i, chunk in enumerate(chunks):
            send_message(phone_number, chunk)
    else:
        send_message(phone_number, chat_response)

    return {"message": "Response sent successfully"}
