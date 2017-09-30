# Tutorial: Loading Python-TensorFlow Trained Model in C++

<p align="center">
<img src="https://github.com/sansinghsanjay/loading_python_tensorflow_model_in_cpp_tensorflow/blob/master/images/logos.png">
</p>

## Introduction
### Python
Python is a widely used programming language for general purpose programming. This programming language is easy to learn and use. Python has been described as readable psuedocode. But the downside of this programming language is that it is not fast and resource efficient, that's why it is called as "Python", slow and space consuming.

Because of its simplicity, it is widely accepted in Machine Learning where professionals from different fields (like Maths, Statistics, Computer Science, etc) collaborate.

As far as research is concern, Python is good for Machine Learning tasks. But for production, where devices are resource constrained and/or it is required to get response from system in real time, Python is a disaster.

In situations like above, the best available solution is C++.

Subsequent sections are focused on loading Python-TensorFlow trained model in C++. In these sections, it is demonstrated that how to store a Python-TensorFlow trained neural network and then how to load it in C++ TensorFlow.

### C++
C++ is also a general purpose programming language. The major advantage of C++ is that it is powerful, fast and resource efficient programming language. Thus, it is able to remove above stated limitations of Python and perfect for resource constrained devices or real time systems.

## Platform Used
This tutorial is tested on Ubuntu-16.04 LTS OS. Mostly, it is observed that softwares which supports Ubuntu-16.04 also supports Ubuntu-14.04 LTS. Thus, this tutorial is also for those who are using Ubuntu-14.04 LTS OS. For rest of people, this may or may not work.

## Required Tools
Following are the tools required for loading Python-TensorFlow trained model in C++:
1. Python
2. Python TensorFlow
3. Bazel
4. C++
5. C++ TensorFlow API

## Installation
1. Python: Following are the commands to install Python with its dependencies and some required Python packages:

```# python dependencies
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
# update packages list
sudo apt-get update
# install python and package management system
sudo apt-get install python2.7
sudo apt-get install python-pip python-dev build-essential
# install python packages
pip install numpy
pip install pandas
pip install scikit-learn
pip install scipy```

You can install other Python packages by using pip.

2. Python TensorFlow: Following are commands to install Python-TensorFlow:
```# install dependency
sudo apt-get install libcupti-dev
# to install python tensorflow cpu
pip install tensorflow
# to install python tensorflow gpu (requires CUDA)
pip install tensorflow-gpu```

3. Bazel: Bazel is an open source software build tool, like cmake and make. Google uses this build tool. This tool will be required for building C++ TensorFlow source code. Instructions for installing bazel are given on: https://docs.bazel.build/versions/master/install-ubuntu.html

4. C++: Run the following command to install C++ in Ubuntu:
```sudo apt-get install g++```

5. C++ TensorFlow API: To build C++ TensorFlow API, follow the instructions till
```sudo cp bazel-bin/tensorflow/libtensorflow_all.so /usr/local/lib```
in step-1 given on:
https://github.com/cjweeks/tensorflow-cmake

## About Python TensorFlow
Python-TensorFlow is a complete library. You can use it for training any kind of neural network. It have all the required functions and it is well documented. Following is the link of Python-TensorFlow document:
https://www.tensorflow.org/api_docs/python/

## About C++ TensorFlow
C++ TensorFlow is still not complete. C++ TensorFlow doesn't have all the functions (like loss functions). So, it is not possible to use C++ TensorFlow for training a network. But it is possible to load a trained TensorFlow model in C++ TensorFlow.

Following is the link of C++ TensorFlow API:
https://www.tensorflow.org/api_guides/cc/guide

This API is not well documented. Few functions documented here are yet not implemented, (as far as I experienced).

That's why, we will train a neural network on MNIST data set and then load that trained model in C++.

## Training And Storing Neural Network in Python-TensorFlow
Python code for training and storing neural network on MNIST data is:

Above script will store entire 
