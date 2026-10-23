
import numpy as np
X=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y=np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
np.random.seed(42)
W1=np.random.randn(4,3)*1.0
b1=np.zeros((1, 4))
W2=np.random.randn(1, 4)*1.0
b2=np.zeros((1,1))
print("W1:",W1.shape)
print("b1:",b1.shape)
print("W2:",W2.shape)
print("b2:",b2.shape)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y, y_pred):
    epsilon = 1e-7
    y_pred =np.clip(y_pred, epsilon,1 - epsilon)
    return -np.mean(
        y * np.log(y_pred)
        + (1 - y) * np.log(1 - y_pred)
    )

def forward_pass(X,W1,b1,W2,b2):
    Z1=X@W1.T+b1
    A1=sigmoid(Z1)

    Z2=A1@W2.T+b2
    A2=sigmoid(Z2)

    return A1, A2


learning_rate=0.1
m=len(X)
for epoch in range(10000):
    A1, A2=forward_pass(X, W1, b1, W2, b2)

    dZ2=A2-y
    dW2=(dZ2.T@A1)/m
    db2=np.sum(dZ2,axis=0,keepdims=True)/m

    dZ1=(dZ2@W2)*A1*(1-A1)
    dW1=(dZ1.T@X)/m
    db1=np.sum(dZ1,axis=0,keepdims=True)/m
    
    W2-=learning_rate*dW2
    b2-=learning_rate*db2
    b1-=learning_rate*db1
    W1-=learning_rate*dW1

Z1 = X @ W1.T + b1
A1 = sigmoid(Z1)
Z2 = A1 @ W2.T + b2
A2 = sigmoid(Z2)
print(binary_cross_entropy(y, A2))
print(A2.ravel())
print(y.ravel())

predictions=list(A2)
for i in range(len(A2)):
    if A2[i]>0.5:
        predictions[i]=1
    else:
        predictions[i]=0
predictions