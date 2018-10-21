---
layout: post
title: "An Introduction to Important Python Libraries"
date: 2018-10-18
excerpt: "A walkthrough of beginner-level Python Libraries."
tags:
- Python
- Pandas
- Numpy
- Libraries
image: "/assets/img/Python.png"
---
![Python]({{"/assets/img/Python.png"}})

As someone who only recently started to learn Python, I know firsthand how overwhelming it can be to try to learn this particular programming language, even though it's supposedly one of the easier ones to learn! Regardless, understanding the power and overall functionality that Python is capable of delivering can certainly instill a greater sense of purpose behind deciding to learn the language. And to understand Python's full-breadth of capabilities, you have to understand how to use some of its most important libraries!

Over the course of this post, I'll cover some of Python's most important libraries, and how you can use them to achieve a variety of tasks. Though this will not be very in-depth for any one library, it'll help you get started on the path to becoming a Python programmer!

## Pandas

![Panda]({{"/assets/img/pandas_cheating.png"}})

Pandas is an open-source library that provides high-performance and easy-to-use data structures and data analysis tools. In my opinion, it's the most important library for a data scientist to know how to use.

To get started, you can install pandas by running this on a command line:
```
pip install pandas
```

Or if you have Anaconda installed already you can run:
```
conda install pandas
```

Or finally, you can run this directly in a jupyter notebook:
```Python
!pip install pandas
```

Once installed, you'll import Pandas into your shell or jupyter notebook by running the following (alias at the end is optional, but commonly used):
```Python
import pandas as pd
```

Here are some important topics to review:
- [Pandas Introduction]('https://pandas.pydata.org/')
- Series - A singular column with x number of rows and a zero-based index.
- DataFrames - Two-dimensional labeled data structures. It's the most important, and most used, way to structure data in Python: [DataFrames]('https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python')

## Numpy

Numpy is another open-source library that adds support for large, multi-dimensional arrays and matrices. It also contains many mathematical functions not found in the standard Python library.

To get started, you can install numpy by running this on a command line:
```
pip install numpy
```

Or if you have Anaconda installed already you can run:
```
conda install numpy
```

Or finally, you can run this directly in a jupyter notebook:
```Python
!pip install numpy
```

Once installed, you'll import Numpy into your shell or jupyter notebook by running the following (alias at the end is optional, but commonly used):
```Python
import numpy as np
```

Here are some important topics to review:
- [Numpy Introduction]('https://docs.scipy.org/doc/numpy-1.15.1/user/quickstart.html')
- Arrays - A data structure that stores values of the same data type (unlike lists, which can store data of any mixed-data types). Probably the most important data structure for scientific computing. The Numpy introduction link above gives a great explanation.
- Matrices - Can be represented using two-dimensional Numpy arrays, and higher dimensional arrays too as needed.
- For a walkthrough of both arrays/matrices: [Arrays & Matrices]('http://www.physics.nyu.edu/pine/pymanual/html/chap3/chap3_arrays.html')

## Matplotlib

Matplotlib is primarily used as a data visualization library, specifically, making 2D plots of arrays in Python. It uses the NumPy library extensively, to the point where it could be said that Matplotlib is built on top of NumPy.

To get started, you can install Matplotlib by running this on a command line:
```
pip install matplotlib
```

Or if you have Anaconda installed already you can run:
```
conda install matplotlib
```

Or finally, you can run this directly in a jupyter notebook:
```Python
!pip install matplotlib
```

Once installed, you'll import Matplotlib into your shell or jupyter notebook by running the following (alias at the end is optional, but commonly used):
```Python
import matplotlib as mpl
```

Different from the other libraries, Matplotlib is most commonly used to import one of it's submodules, PyPlot. It's aim is to try to make matplotlib work like MATLAB. Pyplot functions make various changes to a matplotlib figure - creates a plotting area in a figure, labels the plot, changes the x or y ticks, etc.

You can import Matplotlib.pyplot into your shell or jupyter notebook by running the following (alias at the end is optional, but commonly used):
```Python
import matplotlib.pyplot as plt
```

Here are some important topics to review:
- [Matplotlib Introduction]('https://matplotlib.org/users/intro.html')
- [Matplotlib Wikipedia]('https://en.wikipedia.org/wiki/Matplotlib')
- [Pyplot Introduction]('https://matplotlib.org/users/pyplot_tutorial.html')

That's it for now! This should get you started with Python libraries! Happy learning!
