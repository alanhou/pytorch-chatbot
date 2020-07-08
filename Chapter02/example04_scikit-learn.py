# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
__author__ = 'Alan Hou'

corpus = ["I come to China to travel",
          "This is a car popular in China",
          "I love tea and Apple",
          "This work is to write some papers in science"]

vectorizer = CountVectorizer()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print(tfidf)

"""
  (0, 15)	0.4424621378947393
  (0, 14)	0.697684463383976
  (0, 4)	0.4424621378947393
  (0, 3)	0.348842231691988
  (1, 13)	0.3722248517590162
  (1, 9)	0.47212002654617047
  (1, 6)	0.3722248517590162
  (1, 5)	0.3722248517590162
  (1, 3)	0.3722248517590162
  (1, 2)	0.47212002654617047
  (2, 12)	0.5
  (2, 7)	0.5
  (2, 1)	0.5
  (2, 0)	0.5
  (3, 17)	0.36548060601001114
  (3, 16)	0.36548060601001114
  (3, 14)	0.2881491077345092
  (3, 13)	0.2881491077345092
  (3, 11)	0.36548060601001114
  (3, 10)	0.36548060601001114
  (3, 8)	0.36548060601001114
  (3, 6)	0.2881491077345092
  (3, 5)	0.2881491077345092
"""