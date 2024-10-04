# import numpy as np
#
# # create a random 100x4 matrix
# matrix = np.random.rand(100, 4)
#
# # get the index of the highest element in each row
# max_index = np.argmax(matrix, axis=1)
#
# # transform the matrix to a 100x1 array with the aggregation factor
# result = np.zeros((100, 1))
# for i in range(100):
#     result[i] = matrix[i, max_index[i]]
#
# print(result.shape)
# print(result)


