import matplotlib.pyplot as plt

train_loss = []
train_acc = []
dev_loss = []
dev_acc = []

with open("result.txt", "rb") as f:
    for line in f:
        line = [float(x) for x in line.strip().split()]
        train_loss.append(line[0])
        dev_loss.append(line[1])
        train_acc.append(line[2])
        dev_acc.append(line[3])

x = range(len(train_loss))

plt.plot(x, train_loss, label='train loss',linewidth=2,color='b') 
plt.plot(x, dev_loss, label='dev loss',linewidth=2,color='r') 
plt.xlabel('epoch')
plt.ylabel('acc.')
plt.title('Bowman\'s Model')
plt.legend()
plt.show()