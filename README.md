# AI Agent + WhatsApp Bot


https://github.com/user-attachments/assets/404760ea-4993-4e5c-9c27-37790d5dcedc


## AI Agent
###Set up  and activate python enviroment
    
    - python3 -m venv env
    - source env/bin/activate(on mac)
    -  .\env\Scripts\activate (on windows)

###make installations
    
    - pip install -r requirements.txt

###set up enviroment variables in a file named .env( same folder)
    
    - OPENAI_API_KEY=your open api key
    - if you also want to navigate the option of deploying over a whatsapp sandbox enviroment
    - TWILIO_ACCOUNT_SID #see set up.txt file
    - TWILIO_AUTH_TOKEN #see set up.txt file
    - TWILIO_NUMBER #see set up.txt file

###open and run the python script: test_db; to ensure dummy database (orders.db) is ready and avalaible
    
    - once you see a sample data like below returned, database is good to go
    -.......
    -('SAV_968', 'In Progress', '2025-01-15')
    ('SAV_998', 'Completed', '2025-01-15')

    Order Statuses:
    Completed
    In Progress
    Not Started

###open and run the test_agent.py script to run and test the agent

-  at the current moment, you have several listed messages, go ahead and update the list with messages you might want to pass to the agent
    - test response are to be printed to terminal


## WhatsApp Bot
If you want to demo on whatapp(optional)
read setup.txt file
