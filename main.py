## 数据导入
import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP


data = np.loadtxt(fname='./xor_dataset.csv', delimiter= ',')
# print(data)

### 划分数据
index = np.random.permutation(range(len(data)))
data = data[index]
# print(data)
ratio = 0.8
split = int(ratio * len(data))
x_train, y_train = data[:split,:2], data[:split,2].reshape(-1,1)
x_test, y_test = data[split:, :2], data[split:,2].reshape(-1,1)

###
num_epochs = 200
learning_rate = 0.1
eps = 1e-7
batch_size = 128

mlp = MLP(layer_size=[2,4,1], out_activation='sigmoid')
losses = []
test_losses = []
test_accs = []

for epoch in range(num_epochs):
    st = 0
    loss = 0
    while True:
        end = min(st + batch_size, len(x_train))
        if st > end:
            break
        x = x_train[st:end]
        y = y_train[st:end]
        y_pred = mlp.forward(x)
        grad = (y_pred - y) / (y_pred * (1 - y_pred) + eps)
        mlp.backward(grad)
        mlp.update(learning_rate=learning_rate)
        train_loss = np.sum(- y * np.log(y_pred + eps) - (1-y) * np.log(1-y_pred + eps))
        loss += train_loss
        st += batch_size
    losses.append(loss / len(x_train))
    # 调试：每10个epoch打印一次
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss/len(x_train):.4f}, y_pred range=[{y_pred.min():.4f}, {y_pred.max():.4f}]")
    y_pred = mlp.forward(x_test)
    test_loss = np.sum(- y_test * np.log(y_pred + eps) - (1-y_test) * np.log(1-y_pred + eps))
    test_acc = ((y_pred > 0.5)== y_test).mean()
    test_losses.append(test_loss / len(x_test))
    test_accs.append(test_acc)
    # print(epoch)

print(f'测试准确率: {test_accs[-1]}')
print(losses)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses, label='train_loss')
plt.plot(test_losses, label='test_loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(test_accs, label='test_acc')
plt.legend()
plt.show()
