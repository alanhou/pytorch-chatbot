# -*- coding: utf-8 -*-
import random
from nltk import word_tokenize
__author__ = 'Alan Hou'

# 打招呼
greetings = ['hola', 'hello', 'hi', 'Hi', 'hey', 'hey!']
# 回复打招呼
random_greetings = random.choice(greetings)

# 对于假期的话题关键词
question = ['break', 'holiday', 'vacation', 'weekend']
# 针对假期话题的回答
responses = ['It was nice', 'I went to Paris', 'Sadly, I just stay at home']

# 回复假期话题
random_responses = random.choice(responses)

while True:
    userInput = input(">>> ")
    # 清理输入
    cleaned_input = word_tokenize(userInput)
    # 对比关键词，确定属于哪个问题
    if not set (cleaned_input).isdisjoint(greetings):
        print(random_greetings)
    elif not set (cleaned_input).isdisjoint(question):
        print(random_responses)
    elif userInput == 'bye':
        break
    else:
        print('I did not understand what you said')