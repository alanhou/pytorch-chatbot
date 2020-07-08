# -*- coding: utf-8 -*-
import random
__author__ = 'Alan Hou'

# 打招呼
greetings = ['hola', 'hello', 'hi', 'Hi', 'hey', 'hey!']
# 回复打招呼
random_greetings = random.choice(greetings)

question = ['How are you?', 'How are you doing']
response = ['Okay', 'I\'m fine']

random_response = random.choice(response)

while True:
    user_input = input('>>>')
    if user_input.strip().lower() in greetings:
        print(random_greetings)
    elif user_input in question:
        print(random_response)
    elif user_input == 'bye':
        break
    else:
        print("I did not understand what you said")