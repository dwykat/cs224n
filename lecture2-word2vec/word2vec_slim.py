import tensorflow as tf
import numpy as np
import re
from collections import Counter
import random
import config
import time
import pdb

'''load the data'''
def load_data(filename):
    with open(filename) as f:
        text = tf.compat.as_str(f.read())
    return text

'''preprocess'''
def preprocess(text, freq=5):
    """
     Tokenization/string cleaning for all datasets except for SST.
     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)

    words_raw = text.strip().lower().split()
    word_counts = Counter(words_raw)
    words = [w for w in words_raw if word_counts[w] > freq]
    vocab = set(words)
    vocab2index = {w: idx for idx, w in enumerate(vocab)}
    index2vocab = {idx: w for idx, w in enumerate(vocab)}
    words_int = [vocab2index[w] for w in words]
    return words_int, vocab2index, index2vocab

'''filter words having high frequency but mattering little'''
def resample(words_seq, t=1e-5, threshold=0.8):
    counts = Counter(words_seq)
    num = len(words_seq)
    word_freq = {w: c / num for w, c in counts.items()}
    prob_drop = {w: 1 - np.sqrt(t / word_freq[w]) for w in words_seq}
    train_words = [w for w in words_seq if prob_drop[w] < threshold]
    return train_words

'''get the target given a centor word'''
def get_targets(words, idx, window_size):
    start = idx - window_size if (idx - window_size) > 0 else 0
    end = idx + window_size
    targets = set(words[start: idx] + words[idx+1: end+1])
    return list(targets)

def get_batches(words, batch_size, window_size):
    num_batch = len(words) // batch_size
    words = words[: num_batch * batch_size]
    for i in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[i: i+batch_size]
        for idx in range(len(batch)):
            x_batch = batch[idx]
            y_batch = get_targets(batch, idx, window_size)
            x.extend([x_batch] * len(y_batch))
            y.extend((y_batch))
        yield np.array(x), np.array(y)[:, None]

if __name__ == '__main__':
    filename = config.filename
    filter = config.filter
    data = load_data(filename)

    words_seq, vocab2index, index2vocab = preprocess(data)
    train_seq = resample(words_seq) if filter else words_seq

    embedding_size = config.embedding_size
    vocab_size = len(vocab2index)
    n_sampled = config.n_sampled

    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.int32, [None], name='inputs')
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, inputs)

        softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
        softmax_b = tf.Variable(tf.zeros([vocab_size]))

        # loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
        loss = tf.nn.nce_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        '''evaluate the result by means of the similarity between random words'''

        valid_size = config.valid_size
        valid_window = config.valid_window

        valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
        valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_window), valid_size//2))
        valid_size = len(valid_examples)

        valid_data = tf.constant(valid_examples, dtype=tf.int32)
        # get norm emb
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        norm_embeddings = tf.div(embeddings, norm)

        # compute valid words similarity
        valid_emb = tf.nn.embedding_lookup(norm_embeddings, valid_data) # valid_size * embedding_size
        similarity = tf.matmul(valid_emb, norm_embeddings, adjoint_b=True) # valid_size * vocab_size

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        epochs = config.epochs
        batch_size = config.batch_size
        window_size = config.window_size

        iteration = 1
        loss_count = 0
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs+1):
            batches = get_batches(train_seq, batch_size, window_size)
            start = time.time()
            for x, y in batches:
                train_loss, _ = sess.run([cost, optimizer], feed_dict={inputs: x, labels: y})
                loss_count += train_loss
                if iteration % 100 == 0:
                    end = time.time()
                    print('Epoch: {}/{}'.format(epoch, epochs),
                          'Iteration: {}'.format(iteration),
                          'Avg train loss: {:.4f}'.format(loss_count / 100),
                          '{:.4f} s/batch'.format((end-start)/100))
                    loss_count = 0
                    start = time.time()

                if iteration % 1000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = index2vocab[valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1: top_k+1]
                        print('-------------------------------')
                        print('origin word: {}'.format(valid_word))
                        print('top 8 similar words of it: ')
                        for k in range(top_k):
                            print(index2vocab[nearest[k]])
                        print('-------------------------------')
                iteration += 1

        save_path = saver.save(sess, 'checkpoints/wiki.ckpt')
        embed_mat = sess.run(norm_embeddings)





