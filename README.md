# Machine Learning in Medical Bioinformatics

This repository contains preparatory tasks, reading material and exercises for the [Machine Learning in Medical Bioinformatics](https://canvas.instructure.com/courses/2716790).

## Preparatory tasks (To be finished before the Workshop)
Below is a list of tasks and reading material that should be finished before coming to the Workshop. There is quite a lot of material, so it is completely alright if you don't understand every single detail. The workload for the preparatory tasks should be approximately one week; it is expected that you spend this amount of time to prepare for the physical meeting. How much you spend on the various parts is up to you and your background and interest, but you should come prepared to contribute to the workshop. You are encouraged to bring any questions that have come up for discussion at the workshop.
 
After finishing the preparatory exercise, you should post at least three questions and/or discussion points as answers to this in the [pre-course assignments in canvas](https://canvas.instructure.com/courses/8687227/assignments/45749652) the day before the course starts.  We will spend some time on the first day discussing these questions, so ideally, they should be open-ended. Like I have this data X in my research project, how can I apply machine learning to it to learn trait Y? Even though they could also be simple, like explaining concepts X and Y.


Good luck, and if you have any questions, do not hesitate to contact me. 


### 1. Setup

Many of the exercises will use Jupyter Notebooks, interactive Python environments that make it possible to combine documentation with code. It is also possible to run Python and R code together. Below are some instructions on how to set everything up.

* Read all of the instructions before starting; some tasks are practical, and some are reading, and some might contain some overlap.

* Install [Anaconda](https://www.anaconda.com/download/) on your laptop. This will install a special version of Python that includes the Jupyter Notebook and basically all Python modules needed (deep-learning modules will be installed separately).
<!--, see instructions below under deep learning.  -->

* If you are unfamiliar with Jupyter notebooks you can learn more using the following tutorial: https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook. 

* Git clone this repo: https://github.com/bjornwallner/ML_medbioinfo and make sure you can open the notebook in `notebooks/intro.ipynb` using the following commands:
```
git clone https://github.com/bjornwallner/ML_medbioinfo
cd ML_medbioinfo/notebooks
jupyter-notebook intro.ipynb
```
You can use this notebook or create your own when doing the next exercise.

### 2. PREP: Machine Learning Introduction 
We will use the [scikit-learn](http://scikit-learn.org/stable/index.html) module to do machine learning in Python. It is built on NumPy, SciPy, and matplotlib and is fairly easy to use and it contains all the basic functions to do regular supervised and unsupervised learning. It contains a Neural network module as well, but it is fairly limited, so for neural nets we will use [Tensorflow](http://tensorflow.org) and the [Keras](http://keras.io) API

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


### 3. PREP: Statistical principles in supervised machine learning: overfitting, regularization and all that
Read one of these:

* For those with maths/stats background that want to go slightly deeper into the topic:
  * Hastie et al (2009). [The Elements of Statistical Learning. Springer](https://github.com/bjornwallner/ML_medbioinfo/blob/master/An%20Introduction%20to%20Statistical%20Learning%20%20with%20Applications%20in%20R.pdf).
   <!--https://web.stanford.edu/~hastie/ElemStatLearn/. -->
  Chapters 2 and 3.
* For those with other backgrounds:
  * James et al. (2013). [An Introduction to Statistical Learning – with Applications in R. Springer](https://github.com/bjornwallner/ML_medbioinfo/blob/master/ESLII_print12_toc-2.pdf). Chapters 2 and 6



### 4. PREP: Deep Learning 

Browse or read: Michael Nielsen, Neural Networks and Deep Learning, http://neuralnetworksanddeeplearning.com/ chapter 1-4

<!--
Compute and exercises:

Pointers to exercises might appear here, we are working on updating last year's.


We will use Jupyter Notebooks to run exercises in PyTorch. To get to that you need to install Anaconda and PyTorch  
 
1. __Before course__: Follow the instructions in https://github.com/DeepLearningDTU/02456-deep-learning to install until the section called Native (no need to install TensorFlow). If you already have Anaconda installed from above, just skip this step. 

2. __Before course__: Install PyTorch (https://pytorch.org/). If you followed instruction from above, you have 'conda' package manager and Python version 3.6 installed, you will need to select those to get the PyTorch install command.

3. __Before course__: Run exercises in https://github.com/munkai/pytorch-tutorial/tree/master/1_novice (Do not run exercises involving Cuda)

4. At the course we will run exercises from https://github.com/munkai/pytorch-tutorial/tree/master/2_intermediate

There are also pointers to additional reading material in the notebooks. This is not mandatory.



Research papers - non-mandatory reading

1. Paper https://academic.oup.com/bioinformatics/article/33/21/3387/3931857 and server http://www.cbs.dtu.dk/services/DeepLoc/

2. Paper on Variational auto-encoder for single cell RNA seq analysis by Grønbech et al,  https://www.biorxiv.org/content/early/2018/05/16/318295
-->


### 5. PREP: Project
* Read the project description below
* Think about use cases of ML in your problem domain.


Don't forget that after finishing the preparatory exercise you should post at least three questions and/or discussion points as answers to this [pre-course assignments in canvas](https://canvas.instructure.com/courses/8687227/assignments/45749652), 23:59 day before the course starts at the latest.




----------------------

## Project
The project is the examining part of the course. Together with participation at the workshop it is compulsory to gain the course credits. The workload is expected to be about a week. The project is your chance to learn a bit more about some particular ML methods. The project consists of applying some ML metods to a particular dataset or datasets, and the compare the results. The results should be compiled in a written report including: 

* Description of the chosen methods. In order to compare performance you need either to choose two (or more) different methods or in case of deep learning you could compare different architectures. 
* What parameters are important to optimize for the chosen ML methods
* Which performance measures will be used, correlation, PPV, F1 or AUC? Does it matter?
* Description of your data set.
* Description of how cross-validation was performed. How was the data split to avoid similar examples in training and validation?
* Results from parameter optimizations, plots or tables.
  * What parameters are optimal?
* Conclusions on the difference between ML methods, performance, sensitivity to parameter choices, ease-of-use etc.

If you cannot find a suitable ML project within your particular domain, you can use data from [ProQDock](data/ProQDock.csv), paper: https://academic.oup.com/bioinformatics/article/32/12/i262/2288786. Or you can choose a data set from the [Machine Learning Repository](http://archive.ics.uci.edu/ml/) To study. Make sure it has a good balance between number of examples (# Instances) and number of features (# Attributes).

For example:

* [Epileptic Seizure Recognition Data Set](http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)
* [Mice Protein Expression Data Set](http://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression)


Upload a pdf of your report as answer to the [Project assignment in canvas](https://canvas.instructure.com/courses/6366473/assignments/36123073)





