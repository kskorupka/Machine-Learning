from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.neighbors
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data_breast_cancer = datasets.load_breast_cancer(as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer.data[['mean texture','mean symmetry']], data_breast_cancer.target, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier()
log_reg = LogisticRegression()
knn_clf = sklearn.neighbors.KNeighborsClassifier()

vot_clf_hard = VotingClassifier([('tree_clf',tree_clf),('log_reg',log_reg),('knn_clf',knn_clf)], voting='hard')
vot_clf_soft = VotingClassifier([('tree_clf',tree_clf),('log_reg',log_reg),('knn_clf',knn_clf)], voting='soft')

vot_clf_hard.fit(X_train,y_train)

vot_clf_soft.fit(X_train,y_train)

tree_clf.fit(X_train,y_train)
log_reg.fit(X_train,y_train)
knn_clf.fit(X_train,y_train)

acc_train_tree_clf = accuracy_score(y_train, tree_clf.predict(X_train))
acc_train_log_reg = accuracy_score(y_train, log_reg.predict(X_train))
acc_train_knn_clf = accuracy_score(y_train, knn_clf.predict(X_train))
acc_train_vot_clf_hard = accuracy_score(y_train, vot_clf_hard.predict(X_train))
acc_train_vot_clf_soft = accuracy_score(y_train, vot_clf_soft.predict(X_train))

acc_test_tree_clf = accuracy_score(y_test, tree_clf.predict(X_test))
acc_test_log_reg = accuracy_score(y_test, log_reg.predict(X_test))
acc_test_knn_clf = accuracy_score(y_test, knn_clf.predict(X_test))
acc_test_vot_clf_hard = accuracy_score(y_test, vot_clf_hard.predict(X_test))
acc_test_vot_clf_soft = accuracy_score(y_test, vot_clf_soft.predict(X_test))

acc_vote = [(acc_train_tree_clf, acc_test_tree_clf),(acc_train_log_reg, acc_test_log_reg),(acc_train_knn_clf, acc_test_knn_clf),(acc_train_vot_clf_hard,acc_test_vot_clf_hard),(acc_train_vot_clf_soft,acc_test_vot_clf_soft)]

with open('acc_vote.pkl','wb') as f:
    pickle.dump(acc_vote,f)

with open('acc_vote.pkl','rb') as f:
    test_acc_vote = pickle.load(f)

clf_list = [tree_clf,log_reg,knn_clf,vot_clf_hard,vot_clf_soft]

with open('vote.pkl','wb') as f:
    pickle.dump(clf_list,f)

with open('vote.pkl','rb') as f:
    test_vote_list = pickle.load(f)

bag_clf_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=30)

bag_clf_0_5 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=30, max_samples=0.5)

bag_clf_1.fit(X_train,y_train)

acc_train_bag_clf_1 = accuracy_score(y_train,bag_clf_1.predict(X_train))
acc_test_bag_clf_1 = accuracy_score(y_test,bag_clf_1.predict(X_test))

bag_clf_0_5.fit(X_train,y_train)

acc_train_bag_clf_0_5 = accuracy_score(y_train,bag_clf_0_5.predict(X_train))
acc_test_bag_clf_0_5 = accuracy_score(y_test,bag_clf_0_5.predict(X_test))

past_clf_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=30, bootstrap=False)
past_clf_0_5 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=False)

past_clf_1.fit(X_train,y_train)

acc_train_past_clf_1 = accuracy_score(y_train,past_clf_1.predict(X_train))
acc_test_past_clf_1 = accuracy_score(y_test,past_clf_1.predict(X_test))

past_clf_0_5.fit(X_train,y_train)

acc_train_past_clf_0_5 = accuracy_score(y_train,past_clf_0_5.predict(X_train))
acc_test_past_clf_0_5 = accuracy_score(y_test,past_clf_0_5.predict(X_test))

randfor_clf = RandomForestClassifier(n_estimators=30)

randfor_clf.fit(X_train,y_train)

acc_train_randfor_clf = accuracy_score(y_train,randfor_clf.predict(X_train))
acc_test_randfor_clf = accuracy_score(y_test,randfor_clf.predict(X_test))

ada_clf = AdaBoostClassifier(n_estimators=30)

ada_clf.fit(X_train,y_train)

acc_train_ada_clf = accuracy_score(y_train,ada_clf.predict(X_train))
acc_test_ada_clf = accuracy_score(y_test,ada_clf.predict(X_test))

grad_clf = GradientBoostingClassifier(n_estimators=30)

grad_clf.fit(X_train,y_train)

acc_train_grad_clf = accuracy_score(y_train,grad_clf.predict(X_train))
acc_test_grad_clf = accuracy_score(y_test,grad_clf.predict(X_test))

acc_bag = [(acc_train_bag_clf_1,acc_test_bag_clf_1),(acc_train_bag_clf_0_5,acc_test_bag_clf_0_5),(acc_train_past_clf_1,acc_test_past_clf_1),(acc_train_past_clf_0_5,acc_test_past_clf_0_5),(acc_train_randfor_clf,acc_test_randfor_clf),(acc_train_ada_clf,acc_test_ada_clf),(acc_train_grad_clf,acc_test_grad_clf)]

with open('acc_bag.pkl','wb') as f:
    pickle.dump(acc_bag,f)

with open('acc_bag.pkl','rb') as f:
    acc_bag_test = pickle.load(f)

bag = [bag_clf_1,bag_clf_0_5,past_clf_1,past_clf_0_5,randfor_clf,ada_clf,grad_clf]

with open('bag.pkl','wb') as f:
    pickle.dump(bag,f)

with open('bag.pkl','rb') as f:
    bag_test = pickle.load(f)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(data_breast_cancer.data, data_breast_cancer.target, test_size=0.2, random_state=42)

sampling = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=30, bootstrap=True, bootstrap_features=False,max_features=2,max_samples=0.5)

sampling.fit(X_train_2,y_train_2)

acc_train_samp_clf = accuracy_score(y_train_2,sampling.predict(X_train_2))
acc_test_sampl_clf = accuracy_score(y_test_2,sampling.predict(X_test_2))


samp_list = [acc_train_samp_clf, acc_test_sampl_clf]

with open('acc_fea.pkl','wb') as f:
    pickle.dump(samp_list,f)

with open('acc_fea.pkl','rb')  as f:
    samp_list_test = pickle.load(f)

with open('fea.pkl','wb') as f:
    pickle.dump(sampling,f)

with open('fea.pkl','rb') as f:
    sampling_test = pickle.load(f)

df = pd.DataFrame(columns = ['train_accuracy','test_accuracy', 'features_list'])

for i, estimator in enumerate(sampling.estimators_):
    lista = data_breast_cancer.feature_names[sampling.estimators_features_[i]]
    acc_train = accuracy_score(y_train_2,estimator.predict(X_train_2[lista]))
    acc_test = accuracy_score(y_test_2,estimator.predict(X_test_2[lista]))
    df2 = {'train_accuracy' : acc_train,'test_accuracy' : acc_test, 'features_list' : lista}
    df = df.append(df2,ignore_index=True)

df = df.sort_values(by=['train_accuracy','test_accuracy'],ascending=False)

with open('acc_fea_rank.pkl','wb') as f:
    pickle.dump(df,f)

with open('acc_fea_rank.pkl','rb') as f:
    df_test = pickle.load(f)