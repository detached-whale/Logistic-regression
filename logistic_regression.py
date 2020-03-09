import numpy as np
import matplotlib.pyplot as plt
import mnist_data_loader

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def logistic_regression(x, theta):
    z = np.dot(x, theta)
    return sigmoid(z)

mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=False)

train_set = mnist_dataset.train
test_set = mnist_dataset.test

print("Training dataset size: ", train_set.num_examples)
print("Test dataset size: ", test_set.num_examples)

example_id = 0
image = train_set.images[example_id] # shape = 784 (28*28)
label = train_set.labels[example_id] # shape = 1
plt.imshow(np.reshape(image,[28,28]),cmap='gray')
plt.show()
example_id = 3
image = train_set.images[example_id] # shape = 784 (28*28)
label = train_set.labels[example_id] # shape = 1
plt.imshow(np.reshape(image,[28,28]),cmap='gray')
plt.show()

batch_size = 24
max_epoch = 100
learning_rate = 1e-2

loss_ary = []
acc_ary = []

# initialize theta
theta = np.zeros(train_set.images.shape[1])

for epoch in range(0, max_epoch):
    iter_per_batch = train_set.num_examples // batch_size
    loss = 0
    acc = 0

    for batch_id in range(0, iter_per_batch):
        # get the data of next minibatch (have been shuffled)
        batch = train_set.next_batch(batch_size)
        data, label = batch
        
        # change labels from 3,6 to 0,1
        label = np.where(label == 6, 1, 0)

        # prediction
        predicted = logistic_regression(data, theta)

        # calculate the loss (and accuracy)
        loss += (-label * np.log(predicted) - (1 - label) * np.log(1 - predicted)).mean()
        acc += np.where(np.where(predicted > 0.5, 1, 0) == label, 1 ,0).mean()

        # update weights
        diff = predicted - label
        gradient = np.dot(data.transpose(), diff) / batch_size
        theta -= learning_rate * gradient

    # calculate the loss and the accuracy per epoch
    loss_ary.append(loss/iter_per_batch)
    acc_ary.append(acc/iter_per_batch)


# ploat loss curve
plt.plot(loss_ary)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss curve")
plt.show()

# ploat acc curve
plt.plot(acc_ary)
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("Acc curve")
plt.show()

# predict test test
predicted = logistic_regression(test_set.images, theta)
predicted = np.where(predicted > 0.5, 1, 0)
actual = np.where(test_set.labels == 6, 1, 0)
print("acc of test set = ", np.where(predicted == actual, 1 ,0).mean())
