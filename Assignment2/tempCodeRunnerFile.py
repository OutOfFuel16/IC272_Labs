X1 = df.iloc[:, 0:4].values
# # print(X1)
# C = np.matmul(X1.T, X1)
# print(C)
# eigen = np.linalg.eig(C) # already returns eignvalues (in sorted order) and eigenvectors
# # print(eigen)
# # print(eigen.eigenvectors)
# Q = np.array([eigen[1][:, 0], eigen[1][:, 1]])
# print(Q)
# eigenvalues = eigen[0][:2]
# print(eigenvalues)
# eigenvalues = eigenvalues.reshape(2, 1)
# # print(eigenvalues.shape)
# # print(X1.shape)
# # print(Q.shape)
# X2 = np.matmul(X1, Q.T)
# # print(X2)

# #Q1d

# normalized_eigenvectors = normalize(Q, axis=0)
# scaled_eigenvectors = normalized_eigenvectors * eigenvalues
# # scaled_eigenvectors = Q* eigenvalues
# plt.figure(figsize=(8, 6))
# plt.quiver(0, 0, scaled_eigenvectors[0, 0], scaled_eigenvectors[1, 0], 
#            angles='xy', scale_units='xy', scale=1, color='r', label='PC 1')
# plt.quiver(0, 0, scaled_eigenvectors[0, 1], scaled_eigenvectors[1, 1], 
#            angles='xy', scale_units='xy', scale=1, color='b', label='PC 2')
# plt.scatter(X2[:, 0], X2[:, 1])
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Eigen Directions')
# plt.legend()
# plt.show()

# #Q1e
# X_original = np.matmul(X2, Q)
# # print(X_original)
# rmse = [0, 0, 0, 0]
# for i in range(4):
#     rmse[i] = np.sqrt(np.mean((X_original[:, i] - X1[:, i])**2))
# # print(rmse)

# for i in range(4):
#     print(f'RMSE for feature {attributes[i]}: {rmse[i]}')


# #--------------------------------------------------------------

# #Q2a
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X2, y, test_size=0.2, random_state=104, shuffle=True)
# # print(X_train.shape)
# # print(X_test.shape)
# # print(y_train)
# y_train = y_train.reshape(120, 1)
# y_test = y_test.reshape(1, 30)
# # print(y_train.shape)
# # print(y_test.shape)
# k=0
# y_pred = []
# for i in X_test:
#     # if k==0:
#     #     k+=1
#         dist = []
#         for j in range (0,X_train.shape[0]):
#             dist.append((np.linalg.norm(i-X_train[j]), y_train[j][0]))
#         # print(dist)
#         dist.sort()
#         # print(dist)
#         Iris_setosa = 0
#         Iris_versicolor = 0
#         Iris_virginica = 0
#         for j in range(0,5):
#             if dist[j][1] == 'Iris-setosa':
#                 Iris_setosa+=1
#             elif dist[j][1] == 'Iris-versicolor':
#                 Iris_versicolor+=1
#             else:
#                 Iris_virginica+=1
#         if Iris_setosa > Iris_versicolor and Iris_setosa > Iris_virginica:
#             y_pred.append('Iris-setosa')
#         elif Iris_versicolor > Iris_setosa and Iris_versicolor > Iris_virginica:
#             y_pred.append('Iris-versicolor')
#         else:
#             y_pred.append('Iris-virginica')

# # print(y_pred)
# # print(y_test)

# for i in range(30):
#     if y_pred[i] == y_test[0][i]:
#         k+=1
    
# # Compute the confusion matrix
# cm = confusion_matrix(y_test[0], y_pred, labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# # Create a visual display of the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# # Plot the confusion matrix
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()

