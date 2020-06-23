# 改变padding方式，把前面补齐改为后面补齐
import numpy as np
a = np.array([
    [1,1,1,1,1,2,3,],
    [2,3,4,5,6,7,8,],
    [1,1,2,3,4,5,6,],
])

space_ix = 1
new_a = np.ones_like(a)

index_x, index_y = np.where(a!=space_ix)
num_space = np.sum(a==space_ix, axis=1)