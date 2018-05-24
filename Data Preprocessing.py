
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt


# # Juntar datasets de testes

# In[2]:


import os

mypath = "./data"
test_names = [f for f in os.listdir(mypath) if "test" in f]
test_files = [pd.read_csv("./data/%s"  % file, header = None) for file in test_names]
test = pd.concat(test_files, axis = 0, ignore_index = True)

# export to csv
testfilename = "./data/blogData_test.csv"
test.to_csv(testfilename, index=False, header=False)
print(testfilename + " criado!")


# # Informações do Dataset de treino

# In[3]:


dataset_train = pd.read_csv('data/blogData_train.csv', header=None, names=["X"+str(x) if x != 281 else "Y" for x in range(1, 282)])


# In[4]:


dataset_train.info()


# In[5]:


dataset_train.head()


# In[6]:


dataset_train.describe()


# ## Attribute Information:
# 
# |        Line       |                                                                                                                                  Description                                                                                                                                  |
# |:-----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
# |  **1**...**50**   | Average, standard deviation, min, max and median of the  Attributes 51...60 for the source of the current blog post  With source we mean the blog on which the post appeared.  For example, myblog.blog.org would be the source of  the post myblog.blog.org/post_2010_09_10  |
# |       **51**      |                                                                                                                   Total number of comments *before* basetime                                                                                                                  |
# |       **52**      |                                                                                                         Number of comments in the last 24 hours *before* the basetime                                                                                                         |
# |       **53**      |                                         Let T1 denote the datetime 48 hours *before* basetime,  Let T2 denote the datetime 24 hours *before* basetime.  This attribute is the number of comments in the time period  between T1 and T2                                        |
# |       **54**      |                                                                                    Number of comments in the first 24 hours after the  publication of the blog post, but *before* basetime                                                                                    |
# |       **55**      |                                                                                                                    The diference of attribute *52* and *53*                                                                                                                   |
# |  **56**...**60**  |                                              The same features as the attributes *51*...*55*, but  features *56*...*60* refer to the number of links (trackbacks),  while features *51*...*55* refer to the number of comments.                                               |
# |       **61**      |                                                                                                   The length of time between the publication of the blog post  and basetime                                                                                                   |
# |       **62**      |                                                                                                                          The length of the blog post                                                                                                                          |
# |  **63**...**262** |                                                                                              The 200 bag of *words* features for 200 frequent words of the  text of the blog post                                                                                             |
# | **263**...**269** |                                                                                              binary indicator features (0 or 1) for the weekday (Monday...Sunday) of the basetime                                                                                             |
# | **270**...**276** |                                                                              binary indicator features (0 or 1) for the weekday  (Monday...Sunday) of the date of publication of the blog  post                                                                               |
# |      **277**      |                                                                          Number of parent pages: we consider a blog post P as a  parent of blog post B, if B is a reply (trackback) to  blog post P.                                                                          |
# | **278**...**280** |                                                                                                     Minimum, maximum, average number of comments that the parents received                                                                                                    |
# |      **281**      |                                                                                                The target: the number of comments in the next 24 hours  (relative to basetime)                                                                                                |
#                                                                                                                |

# In[7]:


print(dataset_train.corr()[:]["Y"])


# # Engenharia de características

# In[8]:


from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=(.8 * (1 - .8)))


# In[9]:


from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

rfecv = RFECV(SVR(kernel='linear'))


# In[10]:


from sklearn.feature_selection import SelectKBest, f_regression

selbest = SelectKBest(f_regression,10)


# In[11]:


dataset_test = pd.read_csv('data/blogData_test.csv', header=None, names=["X"+str(x) if x != 281 else "Y" for x in range(1, 282)])


# In[12]:


x_train = dataset_train.drop(dataset_train["Y"])
y_train = dataset_train.iloc[:, -1]

x_train_vt = vt.fit_transform(x_train, y_train)
x_train_rfecv = rfecv.fit_transform(x_train, y_train)
x_train_selbest = selbest.fit_transform(x_train,y_train)


x_test = dataset_test.drop(dataset_test["Y"])
y_test = dataset_test.iloc[:, -1]

x_test_vt = vt.fit_transform(x_test, y_test)
x_test_rfecv = rfecv.fit_transform(x_test, y_test)
x_test_selbest = selbest.fit_transform(x_test,y_test)


# In[ ]:


result = vt.get_support()
restult_rfecv = rfecv.get_support()
result_selbest = selvbest.get_support()

false_indexes = [x+1 for x in range(len(result)) if not result[x]]
print(false_indexes)


# In[ ]:


dataset_new = dataset_train
testset_new = dataset_test
for x in false_indexes:
    dataset_new = dataset_new.drop("X"+str(x), axis=1)
    testset_new = testset_new.drop("X"+str(x), axis=1)


# In[ ]:


print(dataset_new.shape)
print(testset_new.shape)


# In[ ]:


newcsv = "./data/blogData_newTrain.csv"
dataset_new.to_csv(newcsv, index=False, header=False)
print(newcsv + " criado!")

newtestcsv = "./data/blogData_newTest.csv"
testset_new.to_csv(newtestcsv, index=False, header=False)
print(newtestcsv + " criado!")


# In[ ]:


del dataset_new
del testset_new
del dataset_train
del dataset_test


# ## Extra Trees

# In[ ]:


dataset_train = pd.read_csv('data/blogData_train.csv', header=None, names=["X"+str(x) if x != 281 else "Y" for x in range(1, 282)])
dataset_test = pd.read_csv('data/blogData_test.csv', header=None, names=["X"+str(x) if x != 281 else "Y" for x in range(1, 282)])


# In[ ]:


x_train = dataset_train.iloc[:, :-2]
y_train = dataset_train.iloc[:, -1]

print(len(x_train), len(y_train))


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier()
clf.fit(x_train, y_train)


# In[ ]:


model = SelectFromModel(clf, prefit=True)
aux = model.get_support()
false_idx = [x for x in range(len(aux)) if not aux[x]]

dataset_trainnew = dataset_train
dataset_testnew = dataset_test

for x in false_indexes:
    dataset_trainnew = dataset_trainnew.drop("X"+str(x), axis=1)
    dataset_testnew = dataset_testnew.drop("X"+str(x), axis=1)


# In[ ]:


print(dataset_trainnew.shape)
print(dataset_testnew.shape)


# In[ ]:


newcsv = "./data/blogData_newTrainET.csv"
dataset_trainnew.to_csv(newcsv, index=False, header=False)
print(newcsv + " criado!")

newtestcsv = "./data/blogData_newTestET.csv"
dataset_testnew.to_csv(newtestcsv, index=False, header=False)
print(newtestcsv + " criado!")

