# Machine Learning in Medical Bioinformatics

This repository contains preparatory tasks, reading material and exercises for the Machine Learning in Medical Bioinformatics.

## Preparatory tasks (To be finished before the Workshop)
Below is a list of tasks and reading material that should be finished before coming to the Workshop. The workload for the the preparatory tasks should be approximately one week, it is expected that you spend this amount of time to come prepared for the physical meeting. You are encourageFeel free to bring any questions that has come up for to the discussion.


Many of the exercises will use Jupyter notebooks, an interactive Python environments that makes it possible to combine documentation with code. It is also possible to run Python and R code together.

### Setup
* Read all of the instructions before starting, some tasks are practial, e.g. like follow this tutorial and some are reading a few chapters.

* Install [Anaconda](https://www.anaconda.com/download/) on your laptop, choose the Python 3.6 version. This will install a special version of Python that includes the Jupyter Notebook and basically all Python modules needed (deep-learning modules has to be installed separately), see instructions here: https://github.com/DeepLearningDTU/02456-deep-learning.

* If you are unfamiliar with Jupyter notebooks you can learn more using the following tutorial: https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook. 

* Git clone the excerises for the deep learning here: https://github.com/DeepLearningDTU/02456-deep-learning, and make sure that you can open the notebooks.

### PREP: Machine Learning Introduction
We will use the [scikit-learn](http://scikit-learn.org/stable/index.html) module to do machine learning in Python. It is built on NumPy, SciPy, and matplotlib and is fairly easy to use and it contains all the basic functions to do regular supervised and unsupervised learning. It contains a Neural network module as well, but it is fairly limited, so for neural nets we will use [Tensorflow](tensorflow.org) and the [Keras](keras.io) API

* Use a Jupyter notebook to do the first two tutorials on  [scikit-learn](http://scikit-learn.org/stable/tutorial/).
Focusing on the key concepts outlined below. The **goal** with this preparatory exercise is:

  * To understand the key concepts
  * Familiar yourself with Jupyter notebooks

* Actions:

  * Write down descriptions for the concepts
  * Come up with at least three questions that can be discussed.
  * Think about cases were ML can be applied to your particular area (this can be used in the project later as well).
	
Hint: you can hide the prompt and output of code blocks by clicking the top right corner (see below) ![hide_prompt](images/hide_prompt.png)

  1. [An introduction to machine learning with scikit-learn](http://scikit-learn.org/stable/tutorial/basic/tutorial.html)
     * Key concepts:
       * Training set
       * Testing set
       * Samples
       * Features
       * Target
       * Classification
       * Regression
       * Model fit
       * Model predict

  2. [A tutorial on statistical-learning for scientific data processing](http://scikit-learn.org/stable/tutorial/statistical_inference/index.html), stop at "Putting it all together"
     * Key concepts:
       * Supervised learning, incl. examples of methods
       * Unsupervised learning, incl. examples of methods
       * Model selection
       * Model estimator
       * Model parameters
       * Score
       * Cross-validation
       * Grid-search
     


### PREP: Statistical principles in supervised machine learning: overfitting, regularization and all that
Reading list:
* James et al. (2013). An Introduction to Statistical Learning – with Applications in R. Springer. http://www-bcf.usc.edu/~gareth/ISL/. Chapters 2 and 6
* Hastie et al (2009). The Elements of Statistical Learning. Springer. https://web.stanford.edu/~hastie/ElemStatLearn/. Chapters 2 and 3


### PREP: One day mini-course on Deep Learning

Reading material Michael Nielsen, Neural networks and deep learning http://neuralnetworksanddeeplearning.com/

* Chapter 1, [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)
* Chapter 2, [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
* Chapter 3, [Improving the way neural networks learn](http://neuralnetworksanddeeplearning.com/chap3.html)
* Chapter 5, [Why are deep neural networks hard to train?](http://neuralnetworksanddeeplearning.com/chap5.html)
* Chapter 6. [Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)



Chapters 1-3 (stop when reaching section called Overfitting and regularization) and browse Chapter 4. Note that this is a lot of reading material so it is completely alright if you read over the next few weeks. 



----------------------

<!--
## Exercise (During the Workshop)



## Unsupervised learning, theory and exercise.

### Reading

http://scikit-learn.org/stable/tutorial/


## Supervised learning, theory and exercise
### Reading


## One day mini-course on Deep Learning
-->


## Project
The project is the examining part of the course. Together with participation at the workshop it is compulsory to gain the course credits. The workload is expected to be about a week. The project is your chance to learn a bit more about some particular ML methods. Choose at least two supervised and two undersupervised method.

* Describe the chosen methods
* Describe what parameters are important
* Decribe the data sets you have chosen to work with.
  * How do perform cross-validation to avoid over-fitting.



If you cannot find a suitable ML project within your particular domain, choose a data set from the [Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.html) to study. Make sure it has a good balance between number of examples (# Instances) and number of features (# Attributes).

For example:
      * [Epileptic Seizure Recognition Data Set](http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)
      * [Mice Protein Expression Data Set](http://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression)
   