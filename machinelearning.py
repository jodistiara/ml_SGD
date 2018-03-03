import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#DEFINE INITIAL VARIABLES
teta = np.array([0.3,0.6,0.9,0.2])
bias = np.array([0.5])
alpha = 0.8
epoch = 60
error = np.zeros(epoch)
localerror = 0.000000000000000000

#READ DATA
data = pd.read_csv('/home/jodistiara/Documents/machine_learning/iris.data', header=None, nrows=100)

data[4] = data[4].apply(lambda x:str(x).replace('Iris-setosa','0'))
data[4] = data[4].apply(lambda x:str(x).replace('Iris-versicolor','1'))

data[4] = data[4].astype('int64')

#DEFINE FUNCTION
def fungsiH(x, teta, bias):
    sum = bias.copy()
    for i in range(len(x)):
        sum += [x[i]*teta[i]]
    return sum

def sigmoid(ha):
    return (1/(1+math.exp(-1*ha)))

def predict(g):
    if g < 0.5:
        return 0 
    else:
        return 1

def local_error(a,b):
    return math.fabs((a-b))

def delta(g, y, x):
    return (2*(g-y)*(1-g)*g*x)

#TRAIN
for n in range(epoch):
    totalerror = 0
    for i in range(len(data[0])):
        
        x = np.array(data.iloc[i,0:4])
        h = fungsiH(x,teta,bias)
        sigm = sigmoid(h)
        pred = predict(sigm)
        fact = data.iloc[i,4]
        localerror = local_error(sigm, fact)
        print("teta: ", teta)
        print("bias: ", bias)
        dteta = np.zeros(4)
        dbias = np.zeros(1)
        for j in range(len(dteta)):
            dteta[j] = delta(sigm, fact, data.iloc[i,j])
        dbias = np.array(delta(sigm, fact, 1))

        for k in range(len(teta)):
            teta[k] = teta[k] - (alpha*dteta[k])

        bias = bias - (alpha*dbias)

        print("h: ", h)
        print("sigmoid: ", sigm)
        print("predict: ", pred)
        print("fact: ", fact)
        print("local error: ", localerror)
        print("dteta: ", dteta)
        print("dbias: ", dbias)
        print("---------------------------------")
        totalerror = totalerror + localerror

    error[n] = totalerror
    print("error[", n, "]: ", error[n])

plt.clf()

x = np.arange(epoch)
y = error.copy()

plt.plot(x,y)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Grafik Error per Epoch")
plt.show()

