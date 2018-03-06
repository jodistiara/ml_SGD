import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#DEFINE INITIAL VARIABLES
newteta = np.array([0.3,0.6,0.9,0.2])
newbias = np.array([0.5])
teta = newteta.copy()
bias = newbias.copy()
alpha = 0.8
epoch = 60
trainerror = np.zeros(epoch)
testerror = np.zeros(epoch)
localerror = 0.000000000000000000


#READ DATA
data = pd.read_csv('/home/jodistiara/Documents/machine_learning/iris.data', header=None, nrows=100)

data[4] = data[4].apply(lambda x:str(x).replace('Iris-setosa','0'))
data[4] = data[4].apply(lambda x:str(x).replace('Iris-versicolor','1'))

data[4] = data[4].astype('int64')


#SPLIT DATA
traindata = np.vstack([data.iloc[0:40, 0:4], data.iloc[60:100, 0:4]])
trainlabel = np.hstack([data.iloc[0:40, 4], data.iloc[60:100, 4]])

testdata = data.iloc[40:60, 0:4].values
testlabel = data.iloc[40:60, 4].values


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


for n in range(epoch):

    #TRAIN
    totalerror = 0
    for i in range(len(traindata[0])):

        teta = newteta.copy()
        bias = newbias.copy()

        h = fungsiH(traindata[i,:],teta,bias)
        sigm = sigmoid(h)
        pred = predict(sigm)

        localerror = local_error(sigm, trainlabel[i])
        # print("teta: ", teta)
        # print("bias: ", bias)
        dteta = np.zeros(4)
        dbias = np.zeros(1)

        for j in range(len(dteta)):
            dteta[j] = delta(sigm, trainlabel[i], traindata[i,j])
        dbias = np.array(delta(sigm, trainlabel[i], 1))

        for k in range(len(teta)):
            newteta[k] = teta[k] - (alpha*dteta[k])

        newbias = bias - (alpha*dbias)

        # print("h: ", h)
        # print("sigmoid: ", sigm)
        # print("predict: ", pred)
        # print("fact: ", fact)
        # print("local error: ", localerror)
        # print("dteta: ", dteta)
        # print("dbias: ", dbias)
        # print("---------------------------------")
        totalerror = totalerror + localerror

    trainerror[n] = totalerror
    # print("error[", n, "]: ", error[n])

    #TEST
    totalerror = 0
    for i in range(len(testdata[0])):
        
        h = fungsiH(testdata[i,:],newteta,newbias)
        sigm = sigmoid(h)
        pred = predict(sigm)

        localerror = local_error(sigm, testlabel[i])
        # print("teta: ", teta)
        # print("bias: ", bias)
        # print("h: ", h)
        # print("sigmoid: ", sigm)
        # print("predict: ", pred)
        # print("fact: ", fact)
        # print("local error: ", localerror)
        # print("dteta: ", dteta)
        # print("dbias: ", dbias)
        # print("---------------------------------")
        totalerror = totalerror + localerror

    testerror[n] = totalerror


x = np.arange(epoch)
y1 = trainerror.copy()
y2 = testerror.copy()

plt.figure(figsize=(16,8))
plt.plot(x,y1, color="green")
plt.plot(x,y2, color="red")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Grafik Error per Epoch")
plt.show()