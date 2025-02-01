import numpy as np


R = np.array([
    [1, 0.44, 0.29, 0.33],
    [0.44, 1, 0.35, 0.32],
    [0.29, 0.35, 1, 0.60],
    [0.33, 0.32, 0.60, 1]])

eign_val, eign_vector = np.linalg.eig(R)

# 将主成分方差（协方差矩阵特征值）由大到小进行排列
# eign_val_sort = np.sort(eign_val)[::-1]

# 假设要求主成分的累计方差贡献率大于75%
val_sum = np.sum(eign_val)
print((eign_val[0] + eign_val[1]) / val_sum)

lamda1, lamda2 = eign_val[0], eign_val[1]
val1, val2 = eign_vector[0], eign_vector[1]
val = eign_vector[:2]
# print(val.shape) 2, 4




