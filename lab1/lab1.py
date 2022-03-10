import os, urllib.request, tarfile, gzip, shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if not os.path.exists('data'):
    os.mkdir('data')

urllib.request.urlretrieve('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz', 'data/housing.tgz')

file = tarfile.open('data/housing.tgz')
file.extractall('data')

with open('data/housing.csv', 'rb') as f_in:
    with gzip.open('data/housing.csv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
file.close()

os.remove('data/housing.csv')
os.remove('data/housing.tgz')

file = gzip.open('data/housing.csv.gz', 'rb')
for i in range(4):
    print(file.readline())
file.close()


df = pd.read_csv('data/housing.csv.gz')

df.head()

df.info()

df['ocean_proximity'].value_counts()

df['ocean_proximity'].describe()


df.hist(bins=50, figsize=(20,15))
plt.savefig('data/obraz1.png')

df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('data/obraz2.png')

df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('data/obraz3.png')

df.corr()["median_house_value"].sort_values(ascending=False).reset_index(name='wspolczynnik_korelacji').rename(columns={'index' : 'atrybut'}).to_csv('data/korelacja.csv')

sns.pairplot(df)


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
len(train_set),len(test_set)

import pickle 
train_set.to_pickle('data/train_set.pkl')
test_set.to_pickle('data/test_set.pkl')
