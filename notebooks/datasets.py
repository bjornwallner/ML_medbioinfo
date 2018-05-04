
# coding: utf-8

# Fertility Data Set 
# Download: Data Folder, Data Set Description
# 
# Abstract: 100 volunteers provide a semen sample analyzed according to the WHO 2010 criteria. Sperm concentration are related to socio-demographic data, environmental factors, health status, and life habits
# 
# Data Set Characteristics:  
# 
# Multivariate
# 
# Number of Instances:
# 
# 100
# 
# Area:
# 
# Life
# 
# Attribute Characteristics:
# 
# Real
# 
# Number of Attributes:
# 
# 10
# 
# Date Donated
# 
# 2013-01-17
# 
# Associated Tasks:
# 
# Classification, Regression
# 
# Missing Values?
# 
# N/A
# 
# Number of Web Hits:
# 
# 118629
# 
# 
# Source:
# 
# David Gil, 
# dgil '@' dtic.ua.es, 
# Lucentia Research Group, Department of Computer Technology, University of Alicante 
# 
# Jose Luis Girela, 
# girela '@' ua.es, 
# Department of Biotechnology, University of Alicante
# 
# 
# Data Set Information:
# 
# Provide all relevant information about your data set.
# 
# # Attribute Information:
# 
# Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1) 
# 
# Age at the time of analysis. 18-36 (0, 1) 
# 
# Childish diseases (ie , chicken pox, measles, mumps, polio)	1) yes, 2) no. (0, 1) 
# 
# Accident or serious trauma 1) yes, 2) no. (0, 1) 
# 
# Surgical intervention 1) yes, 2) no. (0, 1) 
# 
# High fevers in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1) 
# 
# Frequency of alcohol consumption 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1) 
# 
# Smoking habit 1) never, 2) occasional 3) daily. (-1, 0, 1) 
# 
# Number of hours spent sitting per day ene-16	(0, 1) 
# 
# Output: Diagnosis	normal (N), altered (O)	
# 
# 

# In[1]:

import keras

