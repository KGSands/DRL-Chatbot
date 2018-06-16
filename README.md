# DRL-Chatbot

- CREATE_MODEL:
Build as normal

Change line 21 to folder containing "Converations.csv"

Change line 37 to folder where you want the model stored


- CHATBOT:
Run Training: python chatbot.py -training

Run Testing: python chatbot.py

Change line 61 to the desired checkpoint directory to store the learned Q-values etc

Change line 62 to the desired logs directory for TensorBoard

Change line 65 to the folder containing "Conversations.csv"
