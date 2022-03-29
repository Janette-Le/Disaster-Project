#!/usr/bin/env python
# coding: utf-8

# In[17]:


import sys
#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))
if len(sys.argv)<4:
    print("Enter inputs in followimg format")
    print("#1Enter path of messages file")
    print("#2Enter path for categories file")
    print("#3Enter path for database to be saved")
    quit()
# In[18]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine


# # Functions

# In[19]:


def read_file(path):
    return pd.read_csv(path)


# In[24]:


# load messages dataset
messages = read_file(sys.argv[1]).iloc[:,0:4]


# In[25]:


# load categories dataset
categories = read_file(sys.argv[2])



# In[26]:


# merging both datasets with outer join
df = pd.merge(messages, categories, how="outer")


# In[27]:


# create a dataframe of the 36 individual category columns
categories_splitted = categories["categories"].str.split(";", expand=True)



# In[28]:


# select the first row of the categories dataframe
row = categories["categories"][0].split(";")

# use this row to extract a list of new column names for categories
category_colnames = map(lambda a:a[:-2], row)


# In[9]:


# rename the columns of `categories`
categories_splitted.columns = category_colnames


# In[10]:


for column in categories_splitted:
    # set each value to be the last character of the string
    categories_splitted[column] = list(map(lambda a:a[-1], categories_splitted[column]))

    # convert column from string to numeric
    categories_splitted[column] = list(map(int, categories_splitted[column]))



# In[11]:


# drop the original categories column from `df`
df.drop(["categories"], axis=1, inplace=True)


# In[12]:


categories_splitted["id"] = categories["id"]


# In[13]:


# merging the new categories dataframe with df
df = pd.merge(df, categories_splitted, on="id", how="inner")


# In[14]:


# check number of duplicates

# In[15]:


# drop duplicates
df.drop_duplicates(inplace=True)


# In[16]:


# check number of duplicates


# In[15]:


# Creating SQLite Engine to export DataFrame as DataBase
engine = create_engine(f'sqlite:///{sys.argv[3]}')
df.to_sql('InsertTableName', engine, index=False)


# In[ ]:




