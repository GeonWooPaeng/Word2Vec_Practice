#%%
corpus = ['king is a strong man',
'queen is a wise woman',
'boy is a young man',
'girl is a young woman',
'prince is a young king',
'princess is a young queen',
'man is strong',
'prince is a boy will be king',
'princess is a girl will be queen']

#%%
def remove_stop_words(corpus):
    stop_words = ['is','a','will','be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))

    return results

#%%
corpus = remove_stop_words(corpus)

#%%
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)
words = set(words)


#%%
words

#%%
#Data generation
word2int = {}

for i,word in enumerate(words):
    word2int[word] = i 

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())

WINDOW_SIZE = 2 

data = []
for sentence in sentences:
    for idx,word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence))]:
            if neighbor != word:
                data.append([word,neighbor])

#%%
import pandas as pd 
for text in corpus:
    print(text)
df = pd.DataFrame(data,columns = ['input','label'])

#%%
df.head(10)

#%%
df.shape

#%%
word2int

#%%
#Define Tensorflow Graph
import tensorflow as tf 
import numpy as np 