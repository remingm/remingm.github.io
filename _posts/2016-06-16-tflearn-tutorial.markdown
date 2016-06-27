---
layout: post
title:  "TensorFlow Learn Tutorial - Google's Simple Machine Learning Framework"
date:   2016-06-15 17:58:49 -0700
categories: machine learning TensorFlow neural networks
---

If you are new to machine learning, I recommend first watching [these videos made by Google.](https://youtu.be/cKxRvEZd3Mw?list=PLT6elRN3Aer7ncFlaCz8Zz-4B5cnsrOMt)

In this tutorial, we'll

1. Load our data into pandas dataframes
2. Convert categorical text data into one-hot vectors and numerical vectors
3. Normalize continuous features
4. Split our data into random train and dev sets
5. Run a deep neural network
6. Make a custom model with batch normalization
7. Tune the model with random sampling
8. Log our hyper parameter search to a sortable csv file

For more tutorials and examples, see the [TF Learn home page][tflearnHome].
In this tutorial, all code should be written to one file.

The full code for this tutorial can be found [here][fullcode].

[tflearnHome]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn
[fullcode]:https://github.com/remingm/TF-Learn-Tutorial/blob/master/TFLearn_Tutorial.py


## Part 0 : Get Packages and Data

This tutorial uses the common iris dataset.
[Click here to download the data for this tutorial.][dataDownload]

[dataDownload]:https://github.com/remingm/TF-Learn-Tutorial/archive/master.zip

You will need to install TensorFlow, pandas, and sklearn. Installation instructions can be found at these project's websites. TF Learn is part of TensorFlow. NOTE: This tutorial uses TensorFlow 0.8. As of this writing, TensorFlow 0.9 has [bugs that can complicate using simple models](https://github.com/tensorflow/tensorflow/issues/2727).

# TensorFlow 0.8 Installation:
Install pip (or pip3 for python3) if it is not already installed:

```bash
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev

# Mac OS X
$ sudo easy_install pip
```

Install TensorFlow:

```bash
# Ubuntu/Linux 64-bit, CPU only, Python 2.7:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7. Requires CUDA toolkit 7.5 and cuDNN v4.
# For other versions, see "Install from sources" below.
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
$ sudo easy_install --upgrade six
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl
```

For python3:

```bash
# Ubuntu/Linux 64-bit, CPU only, Python 3.4:
$ sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.4. Requires CUDA toolkit 7.5 and cuDNN v4.
# For other versions, see "Install from sources" below.
$ sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl

# Mac OS X, CPU only:
$ sudo easy_install --upgrade six
$ sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py3-none-any.whl
```



## Part 1 : Load Data

```python
# This tutorial will use TensorFlow, TF Learn, sklearn, and pandas
import random
import numpy
import tensorflow as tf
import tensorflow.contrib.skflow as learn
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import pandas
```

First we'll load our raw data and labels into pandas dataframes

```python
data = pandas.read_csv('iris.data',header=None)
labels = pandas.read_csv('iris.labels',header=None)

# Ignore this. For later one-hot example.
text_labels = labels
```

Let's take a look at our data shapes

```python
print(data.shape)
print(labels.shape)
```

Output:

```bash
(150, 4)
(150, 1)
```

And our raw data

```python
print(data.head())
print(labels.head())
```

Output:

```bash
0    1    2    3   
0  5.1  3.5  1.4  0.2
1  4.9  3.0  1.4  0.2
2  4.7  3.2  1.3  0.2
3  4.6  3.1  1.5  0.2
4  5.0  3.6  1.4  0.2
	  0   
0  Iris-setosa
1  Iris-setosa
2  Iris-setosa
3  Iris-setosa
4  Iris-setosa
```

You can see that each input contains four flower measurements, and each output (label) is a class of the iris plant.

Our model needs to know the number of classes we're predicting

```python
y_classes = len(labels[0].unique()) + 1
```

## Part 2 : Categorical Data Processing

# Representing categories with integers

Our labels are in text format and must be converted to integer values

```python
categorical_processor = learn.preprocessing.CategoricalProcessor()
labels = pandas.DataFrame(categorical_processor.fit_transform(labels.values))
```

Let's look at our new labels. Each category is now a number

```python
print(labels.head())
```

Output:

```bash
0   
0  1
1  1
2  1
3  1
4  1
```

# Representing categories with "one-hot" vectors

Another way to handle categorical text or numbers is converting to categorical one-hot vectors

```python
labels_oneHot = pandas.get_dummies(text_labels)
```
Let's look at our new one-hot labels. Each column now represents a category. A "one-hot" vector is a vector of zeros with a one in a different position for each category.

```python
print(labels_oneHot.head())
```
Output:

```bash
0_Iris-setosa  0_Iris-versicolor  0_Iris-virginica
0            1.0                0.0               0.0
1            1.0                0.0               0.0
2            1.0                0.0               0.0
3            1.0                0.0               0.0
4            1.0                0.0               0.0
```

We won't use our one-hot labels for this experiment, but this is a common machine learning task

## Part 3 : Normalize Continuous Features

It's advantageous to scale continuous features (like our flower measurements) to 0 mean and unit standard deviation.
Note: Don't scale categorical integers.

Let's look at our unscaled data. You'll see that the mean and standard deviation are not 0 and 1 for any columns.

```python
print(data.describe())
```
Output:

```bash
0           1           2           3   
count  150.000000  150.000000  150.000000  150.000000
mean     5.843333    3.054000    3.758667    1.198667
std      0.828066    0.433594    1.764420    0.763161
min      4.300000    2.000000    1.000000    0.100000
25%      5.100000    2.800000    1.600000    0.300000
50%      5.800000    3.000000    4.350000    1.300000
75%      6.400000    3.300000    5.100000    1.800000
max      7.900000    4.400000    6.900000    2.500000
```

First we'll create a scaler

```python
scaler = preprocessing.StandardScaler()
```
Now we'll scale (normalize) our data.

```python
data = scaler.fit_transform(data)
data = pandas.DataFrame(data)
```

Let's look at our scaled data. Now the mean and standard deviation are (roughly) 0 and 1 for each column.

```python
print(data.describe())
```
Output:

```bash
0             1             2             3   
count  1.500000e+02  1.500000e+02  1.500000e+02  1.500000e+02
mean  -4.736952e-16 -6.631732e-16  3.315866e-16 -2.842171e-16
std    1.003350e+00  1.003350e+00  1.003350e+00  1.003350e+00
min   -1.870024e+00 -2.438987e+00 -1.568735e+00 -1.444450e+00
25%   -9.006812e-01 -5.877635e-01 -1.227541e+00 -1.181504e+00
50%   -5.250608e-02 -1.249576e-01  3.362659e-01  1.332259e-01
75%    6.745011e-01  5.692513e-01  7.627586e-01  7.905908e-01
max    2.492019e+00  3.114684e+00  1.786341e+00  1.710902e+00
```

Now we'll split our data into randomly shuffled train, and dev sets.
Setting a random seed allows for continuity between runs

```python
X_train, X_dev, y_train, y_dev = train_test_split(data, labels, test_size=0.2, random_state=42)
```

## Part 4 : Simple Deep Neural Network
Now we'll create a simple deep neural network Tensorflow graph.

```python
classifier = learn.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=y_classes,
							batch_size=32,steps=100, optimizer="Adam",
							learning_rate=0.01, dropout=0.6)
```							 

Here we'll train our DNN

```python
classifier.fit(X_train, y_train, logdir='dnnLogs')
```

Output:

```
Step #100, epoch #25, avg. train loss: 1.23663
```

and evaluate it on our dev data

```python
predictions = classifier.predict(X_dev)
score = metrics.accuracy_score(y_dev, predictions)
print("Accuracy: %f" % score)
```
Output:

```bash
Accuracy: 0.966667
```

Our deep neural network predicted correctly on about 96% of the development set!

## Part 5 : TensorBoard
Running classifier.fit created the directory 'dnnLogs'.
Comment out the following line in part 3 above:

```python
data = scaler.fit_transform(data)
```
Run the DNN again. It will run without scaled features.

Run the following command in a shell from the working directory: `tensorboard --logdir dnnLogs`

Then go to `localhost:6006` in your browser.
Click "Histograms", and then click 'X'.
You'll see two graphs of X. Notice that one is centered around zero and the other isn't. This is what scaling does.

Click "Graph" at the top to see a visual representation of the DNN TensorFlow graph we created.

Uncomment the line you commented out before so the features will be scaled again.

## Part 6 : Custom Model

Next, we'll define a custom classifier model. You can use any TensorFlow or TF Learn code in this function.

```python
def custom_model(X,y):

    # When running, X is a tensor of [batch size, num feats] and y is a tensor of [batch size, num outputs]

    # This model will use a technique called batch normalization
    X = learn.ops.batch_normalize(X, scale_after_normalization=True)

    # Now we'll pass our normalized batch to a DNN
    # We can pass a TensorFlow object as the activation function
    layers = learn.ops.dnn(X,[10,20,10],activation=tf.nn.relu,dropout=0.5)

    # Given encoding of DNN, take encoding of last step (e.g hidden state of the
    # neural network at the last step) and pass it as features for logistic
    # regression over the label classes.
    return learn.models.logistic_regression(layers, y)
```

We need a generic TF Learn model to wrap our custom model.

```python
classifier = learn.TensorFlowEstimator(model_fn=custom_model, n_classes=y_classes,
						   batch_size=32, steps=500,
                                       optimizer="Adam",learning_rate=0.01)
```						   

We'll make a function for training and evaluating

```python
def run_model(classifier,logdir=None,monitor=None):
    # Train
    classifier.fit(X_train, y_train, logdir=logdir,monitor=monitor)

    # Evaluate on dev data
    predictions = classifier.predict(X_dev)
    score = metrics.accuracy_score(y_dev, predictions)
    return score

score = run_model(classifier,'customModelLogs')
print("Accuracy: %f" % score)
```
Output:

```
Step #100, epoch #25, avg. train loss: 0.72944
Step #200, epoch #50, avg. train loss: 0.48746
Step #300, epoch #75, avg. train loss: 0.41292
Step #400, epoch #100, avg. train loss: 0.38321
Step #500, epoch #125, avg. train loss: 0.33969
Accuracy: 1.000000
```

We got 100% accuracy, but keep in mind our dataset has only 150 datapoints.

## Part 7 : Hyperparameter Search
Random sampling is a way to search for the optimal parameters for a model.
It's recommended in 'Bengio. Practical Recommendations for Gradient-Based Training of Deep Architectures. 2012.'

We'll make a function that will randomly generate hyper parameters or return set ones.

```python
def getHyperparameters(tune=False):

    if tune:

        # Randomize DNN layers and hidden size
        hidden_units=[]
        NUNITS = random.randrange(10,100,step=10)
        NLAYERS = random.randint(1,10)
        for layer in range(1,NLAYERS):
            hidden_units.append(NUNITS)

        # Make dict of randomized hyper params
        hyperparams = {
        'BATCH_SIZE':random.randrange(16,28,step=8),
        'STEPS':random.randrange(500,5000,step=100),
        'LEARNING_RATE':random.uniform(0.001, 0.09),
        'OPTIMIZER':random.choice(["SGD", "Adam", "Adagrad"]),
        'HIDDEN_UNITS':hidden_units,
        'NUM_LAYERS':NLAYERS,
        'NUM_UNITS':NUNITS,
        'ACTIVATION_FUNCTION': random.choice([tf.nn.relu, tf.nn.tanh, tf.nn.relu6, tf.nn.elu, tf.nn.sigmoid, tf.nn.softplus]),
        'KEEP_PROB':random.uniform(0.5, 1.0),
        'MAX_BAD_COUNT':random.randrange(10,1000,10)
        }

    else:

        hidden_units=[10,10,10]

        hyperparams = {
        'BATCH_SIZE':32,
        'STEPS':1000,
        'LEARNING_RATE':0.01,
        'OPTIMIZER':"Adam",
        'HIDDEN_UNITS':hidden_units,
        'NUM_LAYERS':len(hidden_units),
        'NUM_UNITS':hidden_units[0],
        'ACTIVATION_FUNCTION': tf.nn.relu,
        'KEEP_PROB':0.6,
        'MAX_BAD_COUNT':random.randrange(10,1000,10)
        }

    return hyperparams
```

Next we'll wrap our model in a function so that we can repeatedly instantiate it with new hyper-parameters.

```python
def instantiateModel(hyperparams):

    # We'll copy the same model from above
    def custom_model(X,y):
        X = learn.ops.batch_normalize(X, scale_after_normalization=True)

        layers = learn.ops.dnn(X, hyperparams['HIDDEN_UNITS'],
	  					activation=hyperparams['ACTIVATION_FUNCTION'],
						dropout=hyperparams['KEEP_PROB'])

        return learn.models.logistic_regression(layers, y)

    classifier = learn.TensorFlowEstimator(model_fn=custom_model, n_classes=y_classes,
						batch_size=hyperparams['BATCH_SIZE'],
						steps=hyperparams['STEPS'],optimizer=hyperparams['OPTIMIZER'],
						learning_rate=hyperparams['LEARNING_RATE'])

    # We'll make a monitor so that we can implement early stopping based on our train accuracy. This will prevent overfitting.
    monitor = learn.monitors.BaseMonitor(early_stopping_rounds=int(hyperparams['MAX_BAD_COUNT']),print_steps=100)

    return classifier, monitor
```

Now we'll 'tune' our model by running a hyper parameter search over many runs

```python
for i in range(100): # Raise this number for a more thorough hyper-parameter search

    hyperparams = getHyperparameters(tune=True)
    print(hyperparams)
    classifier,monitor = instantiateModel(hyperparams)

    score = run_model(classifier,monitor=monitor)
    print("Accuracy: %f" % score)

    # We don't need to log this array
    del hyperparams['HIDDEN_UNITS']

    # Now we'll add the dev set accuracy to our dict
    hyperparams['dev_Accuracy'] = score

    # Convert the dict to a dataframe
    log = pandas.DataFrame(hyperparams,index=[0])
    print(log)

    # Write to a csv file
    csvName = 'model_log.csv'
    if not (os.path.exists(csvName)):
        # First run, write headers
        log.to_csv('model_log.csv', mode='a')
    else:
        log.to_csv('model_log.csv', mode='a',header=False)
```

Open the csv file in libreoffice or a similar editor. Now you can sort thousands of runs by dev accuracy and find the best hyperparameters.

Tuning is important. If you check model_log.csv you'll see that the same model can do poorly or excellently with different hyperparameters.
