---
layout: post
title:  "Commonsense Knowledge Mining from Pretrained Models"
date:   2019-05-01 12:00:00 -0000
categories: ML
---

This is a paper I worked on with Joe Davison's that we presented at EMNLP 2019.

At a high level, we tried to show a computer lots and lots of text and then we figured out a way to extract the commonsense knowledge that the model learned about the world. More specifically, we expressed knowledge as triples (e.g. [clouds, Causes, rain]), developed a model to turn these triples into sentences (e.g. "The clouds caused the rain."), and then ranked a number of these sentences with a metric based on mutual information.

The paper is [here](https://arxiv.org/abs/1909.00505).

A similar method was developed at Facebook AI Research in this [paper](1).

We worked on this project for Sasha Rush's fantastic course on [NLP](https://harvard-ml-courses.github.io/cs287-web/).
