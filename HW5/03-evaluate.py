#!/usr/bin/env python
# coding: utf-8

# The saved models are loaded by re-running 02-train.py

# In[10]:


get_ipython().run_line_magic('run', '02-train.py')


# In[11]:


print('In total, the models do not predict the titles well, the accuracy is 30-40%.')


# In[ ]:


### RUN this only in jupyter notebook
os.system(f'jupyter nbconvert 03-evaluate.ipynb --to python')

