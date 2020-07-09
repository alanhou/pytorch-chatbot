# -*- coding: utf-8 -*-
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
__author__ = 'Alan Hou'

# 构建ChatBot，并指定一个 Adapter
bot = ChatBot(
    'Default Response Example Bot',
    storage_adapter= 'chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand',
            'maximum_similarity_threshold': 0.65
        },
        # {
        #     'import_path': 'chatterbot.logic.LogicAdapter',
        #     'threshold': 0.65,
        #     'default_response': 'I am sorry, but I do not understand'
        # }
    ],
  )

trainer = ListTrainer(bot)

# 手动给定语料用于训练
trainer.train([
    'How can I help you?',
    'I want to create a chat bot',
    'Have you read the documentation?',
    'No, I have not',
    'This should help get your started: https://chatterbot.readthedocs.io/en/latest/quickstart.html'
])

# 给定问题并取回结果
question = 'How do I make an omelette?'
print(question)
response = bot.get_response(question)
print(response)
print("\n")

question = "How to create a chat bot?"
print(question)
response = bot.get_response(question)
print(response)

"""
How do I make an omelette?
I am sorry, but I do not understand


How to create a chat bot?
Have you read the documentation?
"""