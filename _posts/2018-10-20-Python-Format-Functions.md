---
layout: post
title: "Python's Print & Format Functions"
date: 2018-10-20
excerpt: "A walkthrough of Python's Print & Format functions."
tags:
- python
- code along
- format
---
![print_intro]({{"/assets/img/print_quatro.png"}})

Being able to print out your results in Python in a readable and understandable manner is quintessential to being able to interpret your code in a meaningful way. When coding, it's always a best practice to occasionally stop and print your results to make sure the code is working as intended.

In order to better achieve this, using the .format function is invaluable.

It's generally written at the end of a print as such:

```Python
print("Hello {}!".format('Tom'))
```

Will give us the result of:
![print1]({{"/assets/img/print1.png"}})

You can also use it to both calculate and print simple outputs, as such:

```Python
x = 2
y = 3
print("You have successfully added {} to {} and printed {}!".format(x, y, x+y))
```

Gives us an output of:
![print2]({{"/assets/img/print2.png"}})

Alternatively, you can use 'f' before the string to simplify formatting, and triple quotes at the beginning and end to print multiple lines. We even can take this a step further and input functions from other parts of Python to have it print the results. Using data from a previous SAT participation project I worked on:

![print3]({{"/assets/img/complicated_print.png"}})

Print and format are two of the most important functions to know in order to become a good Python programmer!
