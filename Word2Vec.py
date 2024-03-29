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

ONE_HOT_DIM = len(words)

#function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = []
Y = []

for x,y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))

# convert them to numpy arrays
X_train = np.asarray(X)
Y_train = np.asarray(Y)

#making placeholders for X_train and Y_train
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

#word embedding will be 2 dimension for 2d visualization
EMBEDDING_DIM = 2

#hidden layer: which represents word vector eventually
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1])) #bias
hidden_layer = tf.add(tf.matmul(x,W1),b1)

#output layer
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2),b2))

#loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

#training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#%%
#Train
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 20000
for i in range(iteration):
    #input is X_train which is one hot encoded word
    #label is Y_train which is one hot encoded neighbor word
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration' + str(i) + ' loss is : ', sess.run(loss,feed_dict={x:X_train, y_label: Y_train}))

#%%
#Now the hidden layer (W1 + b1) is actually the word look up table
vectors = sess.run(W1 + b1)
print(vectors)


#%%
w2v_df = pd.DataFrame(vectors, columns = ['x1','x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word','x1','x2']]
w2v_df

#%%
#word vector in 2d chart
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word,(x1,x2))

PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

plt.xlim(x_axis_min,x_axis_max)
plt.ylim(y_axis_min,y_axis_max)
plt.rcParams["figure.figsize"] = (10,10)

plt.show()

#%%
