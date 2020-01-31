---
layout: post
title:  "How to Train a Neural Network in 5 Minutes"
date:   2018-11-12 12:00:00 -0000
categories: ML
---
This tutorial will take you from never using python to training your first neural network with some theory along the way!

# To do before the tutorial: Install Python, Jupyter Notebooks, and Keras

The first thing you need to do is install [Anaconda](https://www.anaconda.com/download/). You should choose the latest Python 3 version. Anaconda is a collection of things that make it easy to do data science. Here is what is included in the distribution:
1. **Python**  is an increasingly [popular](https://www.economist.com/science-and-technology/2018/07/19/python-has-brought-computer-programming-to-a-vast-new-audience) programming language for data science.
2. The **Anaconda Navigator**, which lets you launch applications (called integrated development environments or IDEs)  like Jupyter Notebook that help you write Python code.
3. A bunch of useful **packages** (which are basically code that someone else wrote and has "packaged" nicely for you to use) that let you process data quickly and visualize results easily. Some of these packages are downloaded automatically and others you need to download individually yourself.
4. The **conda** package management system that will organize and help you download all of the packages you need.

You can now create a new folder anywhere on your computer to store the files we're going to work on.

Now you should try to open the Anaconda-Navigator application. From there, click on "Launch" underneath the Notebook rectangle. A new tab should open in internet browser that looks like this

![jupyter notebook](JupyterNotebookLanding.png)

Now navigate into the folder you just created and click on `New` in the top right hand corner and then click on `Python 3`. Congratulations! You just created your first Jupyter notebook. Feel free to play around with the notebook and get familiar with Python.

* [Here](https://www.codecademy.com/learn/learn-python) is a python tutorial you can work through if you want.
* [Here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) is a jupyter notebook tutorial, that covers what we've just done and quite a bit more.

A useful shortcut in Jupyter notebook is that `shift + enter` runs the current cell.

Installing everything you need is always the hardest part of any project so don't get discouraged if you run into trouble. When in doubt, google! If that doesn't work, email me at joshua_feldman@g.harvard.edu and I'll try to help.

Great work! See you at the tutorial.

# Who Cares About Deep Learning

With all the hype surrounding deep learning, what neural networks can actually do sometimes gets lost in the noise.

Before we get started, as some inspiration, here are two of the most interesting things I've seen neural networks do. We won't be doing anything like this in the tutorial, but we will be covering the building blocks of these more advanced applications:

This was written by a neural network after being trained on the collective works of Shakespeare. The crazy thing is that the algorithm was never told that words are seperated by spaces and learned, letter by letter, to write.
```
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep.

Second Senator:
They are away this miseries, produced upon my soul,
Breaking and strongly should be buried, when I perish
The earth and thoughts of many states.

DUKE VINCENTIO:
Well, your wit is in the care of side and that.

Second Lord:
They would be ruled after this chamber, and
my fair nues begun out of the fact, to be conveyed,
Whose noble souls I'll have the heart of the wars.

Clown:
Come, sir, I will make did behold your worship.

VIOLA:
I'll drink it.
```

In addition to writing Shakespeare, deep learning is making it easier to generate all sorts of [media](https://www.youtube.com/watch?v=cQ54GDm1eL0).

# But What is a Neural Network?

I would explain, but [this video](https://www.youtube.com/watch?v=aircAruvnKk&t) does it better than I ever could. 3Blue1Brown puts out the best videos on the internet. It's a 4 part series, but we're only going to cover the first one.

# Install Tensorflow

Follow the instruction [here](https://www.tensorflow.org/install/pip). This is definitely be the hardest part.

# Time to code!

We're going to be working from this [introduction](https://keras.io/) to Keras. Keras is a package that let's you build neural networks super easily.

Our goal will be to classify the handwritten digits we saw in the video. This is actually a famous machine learning dataset called MNIST. We can download it directly from keras.


```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

What we just downloaded are as follows:

`x_train` - 60000 images represented by 28x28 pixels. We will use this data to train our network.

`y_train` - 60000 labels represented as a single number between 0 and 9. We will use this data to train our network.

`x_test` - 10000 images represented by 28x28 pixels. We will use this data to evaluate our network.

`y_test` - 10000 labels represented as a single number between 0 and 9. We will use this data to evaluate our network.

The reason why we have seperate training and test data is because it would give us an unfair advantage if we only evaluated our neural network on examples we used for training. It would be like telling someone that 2 + 2 = 4 and then asking them to tell us what 2 + 2 equals. If they said 4, would that mean they understood addition or would they just be memorizing what we told them?


We can take a look at one of our images just as a sanity check.


```python
import matplotlib.pyplot as plt # a plotting library
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
print('this is labelled as a {}'.format(y_train[0]))
```

    this is labelled as a 5



![png](/images/output_25_1.png)


We need to preprocess our data a bit.


```python
from keras.utils import np_utils
# get number of pixels
num_pixels = 28 * 28

# change training images from a 28x28 grid to a row or 728 numbers
x_train = x_train.reshape(60000, 728)

# do the same thing for the test images
x_test = x_test.reshape(60000, 10)

#scale the image from a 0-225 range to a 0-1 range
x_train = x_train/225

#do the same thing for the test images
x_test = x_test/225

# we need to change our labels to a "one hot" encoding
y_train = np_utils.to_categorical(y_train)

#do the same thing for the test labels
y_test = np_utils.to_categorical(y_test)
```

The first thing we're going to do is create a `model`. A model is the basic object we're going to work with.


```python
from keras.models import Sequential

model = Sequential()
```

The `Sequential` command just creates an empty neural network. It doesn't contain any layers.

We are now going to recreate the neural network from the 3blue1brown video. To do so, we add two dense layers. There are many different types of layers that can go in a neural network, but the simplest is the one we saw in the video where every neurom in the previous layer is connected to every neuron in the current layer. We go from 784 input neurons to 16 hidden neurons to 10 output neurons corresponding to the probability of the image being one of the 10 digits.


```python
from keras.layers import Dense

model.add(Dense(units=16, activation='sigmoid', input_dim=784)) # add the first layer of the network
model.add(Dense(units=10, activation='softmax'))
```

We can see our full model with the `summary` command.


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_13 (Dense)             (None, 16)                12560     
    _________________________________________________________________
    dense_14 (Dense)             (None, 10)                170       
    =================================================================
    Total params: 12,730
    Trainable params: 12,730
    Non-trainable params: 0
    _________________________________________________________________


We didn't go into exactly what this next bit of code is doing, but basically it specifies how our neural network is going to learn. We can go into this more later if there is time. You can also watch more of the 3blue1brown videos and it will give you a better idea.


```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

We are now ready to train the model! We call the `model.fit` command and pass the following parameters.

`x_train` - our training images

`y_train` - our labels

`epochs` - how many times our neural network is going to see our full dataset

`batch_size` - instead of training our network on all our data at once, we're going to only show it smaller "batches" of 32 images at time.




```python
model.fit(x_train, y_train, epochs=20, batch_size=32)
```

    Epoch 1/20
    60000/60000 [==============================] - 2s 32us/step - loss: 0.3790 - acc: 0.9002
    Epoch 2/20
    60000/60000 [==============================] - 2s 30us/step - loss: 0.3649 - acc: 0.9026
    Epoch 3/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.3528 - acc: 0.9056
    Epoch 4/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.3422 - acc: 0.9085: 0s - loss: 0.3429 - acc:
    Epoch 5/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.3330 - acc: 0.9107
    Epoch 6/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.3247 - acc: 0.9126
    Epoch 7/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.3173 - acc: 0.9147
    Epoch 8/20
    60000/60000 [==============================] - 2s 30us/step - loss: 0.3106 - acc: 0.9156
    Epoch 9/20
    60000/60000 [==============================] - 2s 31us/step - loss: 0.3045 - acc: 0.9172
    Epoch 10/20
    60000/60000 [==============================] - 2s 31us/step - loss: 0.2989 - acc: 0.9188
    Epoch 11/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.2937 - acc: 0.9198: 1s -
    Epoch 12/20
    60000/60000 [==============================] - 2s 30us/step - loss: 0.2889 - acc: 0.9213
    Epoch 13/20
    60000/60000 [==============================] - 2s 30us/step - loss: 0.2844 - acc: 0.9226
    Epoch 14/20
    60000/60000 [==============================] - 2s 31us/step - loss: 0.2802 - acc: 0.9235
    Epoch 15/20
    60000/60000 [==============================] - 2s 32us/step - loss: 0.2762 - acc: 0.9246
    Epoch 16/20
    60000/60000 [==============================] - 2s 30us/step - loss: 0.2726 - acc: 0.9254: 0s - loss:
    Epoch 17/20
    60000/60000 [==============================] - 2s 32us/step - loss: 0.2691 - acc: 0.9262
    Epoch 18/20
    60000/60000 [==============================] - 2s 37us/step - loss: 0.2657 - acc: 0.9269
    Epoch 19/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.2626 - acc: 0.9275
    Epoch 20/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.2596 - acc: 0.9285





    <keras.callbacks.History at 0x14a9f60b8>



To evaluate the model, we call `model.evaluate` on our test data. The second number in the output is our accuracy.


```python
model.evaluate(x_test, y_test)
```

    10000/10000 [==============================] - 0s 18us/step





    [0.2570587786972523, 0.9297]



As a sanity check, we can make predictions on our test data and match them up with our labels. We need to convert the probabilities returned by our neural network back to labels.


```python
classes = model.predict(x_test, batch_size=128)
```


```python
import numpy as np
predictions = np.argmax(classes, axis = 1)
```


```python
labels = np.argmax(y_test, axis = 1)
```


```python
sum(predictions == labels)/len(labels)
```




    0.9297



We got 92% accuracy! More importantly, you just trained your first neural network!
