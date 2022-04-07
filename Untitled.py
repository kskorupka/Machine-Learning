#!/usr/bin/env python
# coding: utf-8

# In[69]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer.data[['mean texture','mean symmetry']], data_breast_cancer.target, test_size=0.2, random_state=42)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.neighbors


# In[30]:


tree_clf = DecisionTreeClassifier()


# In[31]:


log_reg = LogisticRegression()


# In[34]:


knn_clf = sklearn.neighbors.KNeighborsClassifier()


# In[35]:


from sklearn.ensemble import VotingClassifier


# In[45]:


vot_clf_hard = VotingClassifier([('tree_clf',tree_clf),('log_reg',log_reg),('knn_clf',knn_clf)], voting='hard')


# In[47]:


vot_clf_soft = VotingClassifier([('tree_clf',tree_clf),('log_reg',log_reg),('knn_clf',knn_clf)], voting='soft')


# In[46]:


vot_clf_hard.fit(X_train,y_train)


# In[48]:


vot_clf_soft.fit(X_train,y_train)


# In[49]:


tree_clf.fit(X_train,y_train)
log_reg.fit(X_train,y_train)
knn_clf.fit(X_train,y_train)


# In[59]:


acc_train_tree_clf = tree_clf.score(X_train,y_train)
acc_train_log_reg = log_reg.score(X_train,y_train)
acc_train_knn_clf = knn_clf.score(X_train,y_train)
acc_train_vot_clf_hard = vot_clf_hard.score(X_train,y_train)
acc_train_vot_clf_soft = vot_clf_soft.score(X_train,y_train)


# In[60]:


acc_test_tree_clf = tree_clf.score(X_test,y_test)
acc_test_log_reg = log_reg.score(X_test,y_test)
acc_test_knn_clf = knn_clf.score(X_test,y_test)
acc_test_vot_clf_hard = vot_clf_hard.score(X_test,y_test)
acc_test_vot_clf_soft = vot_clf_soft.score(X_test,y_test)


# In[63]:


acc_vote = [(acc_train_tree_clf, acc_test_tree_clf),(acc_train_log_reg, acc_test_log_reg),(acc_train_knn_clf, acc_test_knn_clf),(acc_train_vot_clf_hard,acc_test_vot_clf_hard),(acc_train_vot_clf_soft,acc_test_vot_clf_soft)]


# In[68]:


print(acc_train_tree_clf)
print(acc_test_tree_clf)
print(acc_train_log_reg)
print(acc_test_log_reg)
print(acc_train_knn_clf)
print(acc_test_knn_clf)
print(acc_train_vot_clf_hard)
print(acc_test_vot_clf_hard)
print(acc_train_vot_clf_soft)
print(acc_test_vot_clf_soft)


# In[67]:


for x in acc_vote:
    for y in x:
        print(y)


# In[70]:


with open('acc_vote.pkl','wb') as f:
    pickle.dump(acc_vote,f)


# In[71]:


with open('acc_vote.pkl','rb') as f:
    test_acc_vote = pickle.load(f)


# In[73]:


for x in test_acc_vote:
    for y in x:
        print(y)


# In[74]:


clf_list = [tree_clf,log_reg,knn_clf,vot_clf_hard,vot_clf_soft]


# In[75]:


with open('vote.pkl','wb') as f:
    pickle.dump(clf_list,f)


# In[76]:


with open('vote.pkl','rb') as f:
    test_vote_list = pickle.load(f)


# In[110]:


for x in test_vote_list:
    print(type(x))


# In[111]:


from sklearn.ensemble import BaggingClassifier


# In[112]:


bag_clf_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=30)


# In[113]:


bag_clf_0_5 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=30, max_samples=0.5)


# In[114]:


bag_clf_1.fit(X_train,y_train)


# In[115]:


acc_train_bag_clf_1 = bag_clf_1.score(X_train,y_train)
acc_test_bag_clf_1 = bag_clf_1.score(X_test,y_test)


# In[116]:


print(acc_train_bag_clf_1, acc_test_bag_clf_1)


# In[117]:


bag_clf_0_5.fit(X_train,y_train)


# In[118]:


acc_train_bag_clf_0_5 = bag_clf_0_5.score(X_train,y_train)
acc_test_bag_clf_0_5 = bag_clf_0_5.score(X_test,y_test)


# In[119]:


print(acc_train_bag_clf_0_5, acc_test_bag_clf_0_5)

