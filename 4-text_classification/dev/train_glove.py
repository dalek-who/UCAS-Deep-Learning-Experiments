#%%
# 训练Glove词向量
from glove import Glove
from glove import Corpus

# 读取训练数据。先转换成Corpus形式
sentense = []
with open("poem_for_embedding.txt") as f:
    for line in f.readlines():
        sentense.append(line.replace("\n", "").split(" "))
corpus_model = Corpus()
corpus_model.fit(sentense, window=5)  # window： 滑动窗口大小

# 训练glove
embedding_dim = 10
glove = Glove(no_components=embedding_dim, learning_rate=0.05)  # no_components：词嵌入维度，
glove.fit(corpus_model.matrix, epochs=10, no_threads=4, verbose=True)  # verbose：训练时是否打印info
glove.add_dictionary(corpus_model.dictionary)

glove.save(f'glove_{embedding_dim}.txt')

# glove = Glove.load(f'glove_{embedding_dim}.txt')
# glove.most_similar('我', number=10)