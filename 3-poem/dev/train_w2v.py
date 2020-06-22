# 读取训练数据。先转换成Corpus形式
from gensim.models import word2vec
from gensim.models import KeyedVectors, Word2Vec

sentences = []
with open("poem_for_embedding.txt") as f:
    for line in f.readlines():
        sentences.append(line.replace("\n", "").split(" "))

dim=128
window=5
min_count=5
model=word2vec.Word2Vec(sentences, size=dim, window=window, min_count=min_count, workers=4)
model.save(f"vocab/w2v_{dim}.txt")
print(model.wv.most_similar("我", topn=10))
print(model.wv["我"])
