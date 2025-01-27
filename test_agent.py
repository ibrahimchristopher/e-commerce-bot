######THE EASIEST WAY TO TEST OUT THE SYSTEM AFTER EVERYTHING HAS BEEN SET UP#########
##PLEASE READ THROUGH

from agent import AgenticSystem
from groq import Groq

agent = AgenticSystem()

thread_id = 'test_id' #Using this to keep track of conversation context

config = {
    "configurable": {"thread_id": thread_id,}}

def respond_to_query(query):
    response = agent.query(query=query, config=config, last_message_only=False)
    chat_response = response[-1].content
    print(chat_response)

test_questions = [
    "Hi there, what is the status for order SAV_5",
    "Hi there, what is the status for order SAV_190",
    "What is the return policy for items purchased at our store?",
    "Are there any items that cannot be returned under this policy?",
    "How will I receive my refund?",
    "i want to speak to a rep",
    "my name is Christopher, chris@gmail.com, 08154481095",
    "hi my order is getting delayed, i want to talk to a somone about this",
    "my name is kendrick lamar, kdot@gmail.com, 08154481098"
]

for test_q in test_questions:
    print(test_q)
    respond_to_query(test_q)
    print("_____"*20)



#lets really take it up a notch here, make the customers ruder, less direct and morelike the rea world
advanced_questions = [
    "Hi is my package status of order SAV_21 ever going to be completed or are the workers asleep",#rude
    "i purchased some watches last week can i still return them",#less direct 
    "how about apples",#deviating from policy
    "i paid by card but can i get a cash refund and i dont feel comfortable talking to a non human",##deviating from policy and indirect request to talk to rep
    "my name is jack harlow, jack@gmail.com, 08154481095",
    ]

##testing the advanced questions

for test_q in advanced_questions:
    print(test_q)
    respond_to_query(test_q)
    print("_____"*20)
