# Machine Learning in Medical Bioinformatics

This repository contains preparatory tasks, reading material and exercises for the Machine Learning in Medical Bioinformatics.

## Preparatory tasks (To be finished before the Workshop)
Below is a list of tasks and reading material that should be finished before coming to the Workshop. There are quite a lot of material so it is completely alright if you don't understand every single detail. The workload for the the preparatory tasks should be approximately one week, it is expected that you spend this amount of time to be prepared for the physical meeting. How much you spend on the various parts is up to you and your specific background and interest, but you should come prepared and contribute to the workshop. You are encouraged to bring any questions that has come up for discussion at the workshop.
 

After finishing (or during) the preparatory exercise you should post at least three questions and/or discussion points at the [canvas discussion forum](https://canvas.instructure.com/courses/1308611/discussion_topics/6553199) before June 8, 23:59:00.

Good luck and if you have any question, do not hesitate to contact me. Contact details [here](https://canvas.instructure.com/courses/1308611/users/14364140)


### 1. Setup

Many of the exercises will use Jupyter notebooks, an interactive Python environments that makes it possible to combine documentation with code. It is also possible to run Python and R code together. Below are som instructions on how to set everything up.

* Read all of the instructions before starting, some tasks are practial, and some are reading, and some might contain some overlap.

* Install [Anaconda](https://www.anaconda.com/download/) on your laptop, choose the Python 3.6 version. This will install a special version of Python that includes the Jupyter Notebook and basically all Python modules needed (deep-learning modules has to be installed separately), see instructions below under deep learning. 

* If you are unfamiliar with Jupyter notebooks you can learn more using the following tutorial: https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook. 

* Git clone this repo: https://github.com/bjornwallner/ML_medbioinfo and make sure you can open the notebook in `notebooks/intro.ipynb`
```
git clone https://github.com/bjornwallner/ML_medbioinfo
cd ML_medbioinfo/notebooks
jupyter-notebook intro.ipynb

```

### 2. PREP: Machine Learning Introduction 
We will use the [scikit-learn](http://scikit-learn.org/stable/index.html) module to do machine learning in Python. It is built on NumPy, SciPy, and matplotlib and is fairly easy to use and it contains all the basic functions to do regular supervised and unsupervised learning. It contains a Neural network module as well, but it is fairly limited, so for neural nets we will use [Tensorflow](tensorflow.org) and the [Keras](keras.io) API

* Use a Jupyter notebook to do the first two tutorials on  [scikit-learn](http://scikit-learn.org/stable/tutorial/).
Focusing on the key concepts outlined below. The **goal** with this preparatory exercise is:

  * To understand the key concepts
  * Familiar yourself with Jupyter notebooks

* Actions:

  * Write down descriptions for the concepts
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


### 3. PREP: Disease modelling - Ole Christian Lingjærde 
Reading material: might appear here.

### 4. PREP: Statistical principles in supervised machine learning: overfitting, regularization and all that
Read one of these:

* For those with maths/stats background that want to go slightly deeper into the topic:
  * Hastie et al (2009). The Elements of Statistical Learning. Springer. https://web.stanford.edu/~hastie/ElemStatLearn/. Chapters 2 and 3
* For those with other backgrounds:
  * James et al. (2013). An Introduction to Statistical Learning – with Applications in R. Springer. http://www-bcf.usc.edu/~gareth/ISL/. Chapters 2 and 6



### 5. PREP: One day mini-course on Deep Learning

Reading material: 

__Before course__ browse or read: Michael Nielsen, Neural Networks and Deep Learning, http://neuralnetworksanddeeplearning.com/ chapter 1-4

Compute and exercises: 

We will use Jupyter Notebooks to run exercises in PyTorch. To get to that you need to install Anaconda and PyTorch  
 
1. __Before course__: Follow the instructions in https://github.com/DeepLearningDTU/02456-deep-learning to install until the section called Native (no need to install TensorFlow).

2. __Before course__: Install PyTorch (https://pytorch.org/) 

3. __Before course__: Run exercises in https://github.com/munkai/pytorch-tutorial/tree/master/1_novice (Do not run exercises involving Cuda)

4. At the course we will run exercises from https://github.com/munkai/pytorch-tutorial/tree/master/2_intermediate

There are also pointers to additional reading material in the notebooks. This is not mandatory.

Research papers - non-mandatory reading

1. Paper https://academic.oup.com/bioinformatics/article/33/21/3387/3931857 and server http://www.cbs.dtu.dk/services/DeepLoc/

2. Paper on Variational auto-encoder for single cell RNA seq analysis by Grønbech et al, to appear.


----------------------

## Project
The project is the examining part of the course. Together with participation at the workshop it is compulsory to gain the course credits. The workload is expected to be about a week. The project is your chance to learn a bit more about some particular ML methods. The project consists of applying some ML metods to a particular dataset or datasets, and the compare the results. The results should be compiled in a written report, involving:

* Description of the chosen methods. In order to compare performance you need either to chose two different methods or in case of deep learning you could compare different architectures.
* Description on what parameters are important
* Description of the data sets you have chosen to work with.
  * How cross-validation was performed to avoid over-fitting.
* Presentation of the results and conclusions.

If you cannot find a suitable ML project within your particular domain, choose a data set from the [Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.html) to study. Make sure it has a good balance between number of examples (# Instances) and number of features (# Attributes).

For example:

* [Epileptic Seizure Recognition Data Set](http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)
* [Mice Protein Expression Data Set](http://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression)
* Some more will be added, if you happen to have a good data set (good meaning tabular data with features+target) laying around that you think others could use, feel free to share. 


### 6. PREP: Project
* Think about use cases of ML in your problem domain.