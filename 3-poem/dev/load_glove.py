from glove import Glove
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors, Word2Vec

embedding_dim = 10
glove = Glove.load(f'glove_{embedding_dim}.txt')
print(glove.most_similar('我', number=10))


glove_file_path = f"/data/users/wangyuanzheng/projects/ucas_DL/3-poem/dev/glove_{embedding_dim}.txt"
word2vec_output_path = f"/data/users/wangyuanzheng/projects/ucas_DL/3-poem/dev/glove_{embedding_dim}_w2v.txt"
# 输入文件
glove_file = datapath(glove_file_path)
# 输出文件
tmp_file = get_tmpfile(word2vec_output_path)

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>

# 开始转换
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)

# 加载转化后的文件
model = KeyedVectors.load_word2vec_format(tmp_file, unicode_errors='ignore')