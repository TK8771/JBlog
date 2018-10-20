---
layout: post
title: "A Brief Introduction to Random Forest"
date: 2018-10-16
excerpt: "A walkthrough of Random Forest and the concepts that underlie it."
tags:
- random forest
- decision trees
- classifiers
- models
- blog
image: "/assets/img/random_forest.jpeg"
---
![Tree]({{"/assets/img/random_forest.jpeg"}})
**Let's take a random walk down a Random Forest path!**

Scientists have always been looking to explain the world through observations and data. But what has always fascinated me about life, is that randomness, one way or another, will always play its part.

At least that's what I'd like to think Tim Kam Ho, a computer scientist at IBM Watson, had in mind when she first created Random Forest.

This blog post will cover the basics of the Random Forest model and the concepts that underlie it.

## Decision Trees

A decision tree is a flow-chart-like tree structure, where each internal node represents a test/question on an attribute, each branch represents an outcome of the test/question, and leaf nodes represent different class breakdowns.

Here's an example of a decision tree surrounding the question I hope I'll soon be considering: "Should I accept a new job offer?"

![Decision Tree]({{"/assets/img/decision_tree.png"}})

## Explaining Information Gain - Gini and Entropy

$$ \text{Entropy} = -\sum_{i=1}^{classes} p(i\;|\;t) \;log_2( p(i\;|\;t) ) $$

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

{% highlight ca65 %}
$$ \text{Entropy} = -\sum_{i=1}^{classes} p(i\;|\;t) \;log_2( p(i\;|\;t) ) $$
{% endhighlight %}
