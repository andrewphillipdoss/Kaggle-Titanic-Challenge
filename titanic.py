

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Found this Idea online: group NaN ages by the passenger's Title
def estimate_age(df):
    df['Title']=0
    for data in df:
        df['Title']=df.Name.str.extract('([A-Za-z]+)\.')

    #Assumed Military Titles, Rev, and Dr to be Mr Because of when the Titanic took place, not Sexism :)
    df['Title'].replace(['Mlle', 'Mme', 'Ms', 'Lady', 'Countess', 'Major', 'Don', 'Col', 'Capt', 'Jonkheer', 'Sir', 'Rev', 'Dr'],
                        ['Miss', 'Miss', 'Miss', 'Mrs', 'Mrs', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr'],inplace=True)

    #Replace Null Age Values with predicted Age Values
    df.loc[(train_df.Age.isnull())&(train_df.Title=='Mr'), 'Age']=33.02
    df.loc[(train_df.Age.isnull())&(train_df.Title=='Mrs'), 'Age']=35.98
    df.loc[(train_df.Age.isnull())&(train_df.Title=='Miss'), 'Age']=21.86
    df.loc[(train_df.Age.isnull())&(train_df.Title=='Master'), 'Age']=4.57

#Age is a continuous feature, and Machine Learning Models don't do well with those
#create an Age_group to classify by
def create_band(df, series, band_name):
    df[band_name]=0
    df.loc[df[series]<=16,band_name]=0
    df.loc[(df[series]>16)&(df[series]<=32),band_name]=1
    df.loc[(df[series]>32)&(df[series]<=48),band_name]=2
    df.loc[(df[series]>48)&(df[series]<=64),band_name]=3
    df.loc[df[series]>64,band_name]=4

#SibSp and Parch refer to how many siblings/spouses or parents are on board
#We'll use this function to change this to a binary isAlone
def is_alone(col):
   if col['SibSp'] == 0 & col['Parch'] == 0:
      return 1
   else:
      return 0

#Read CSVs
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

#estimate ages
estimate_age(train_df)
estimate_age(test_df)
#create age bands
create_band(train_df, 'Age', 'AgeBand')
create_band(test_df, 'Age', 'AgeBand')

#create isAlone
train_df['isAlone'] = train_df.apply(is_alone, axis=1)
test_df['isAlone'] = train_df.apply(is_alone, axis=1)



#Init and Choose Label for Classification
le = LabelEncoder()
labels = np.asarray(train_df.Survived)
le.fit(labels)

#Select Train Features
train_df_selected = train_df.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'Age'], axis=1)
train_df_features = train_df_selected.to_dict(orient='records')

#Select Test Features
test_df_selected = test_df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'Age'], axis=1)
test_df_features = test_df_selected.to_dict(orient='records')

#Prepare Test and Train Features
vec = DictVectorizer()
train_features = vec.fit_transform(train_df_features).toarray()
test_features = vec.fit_transform(test_df_features).toarray()

#Train Classifier
#clf = RandomForestClassifier(min_samples_split=4)
clf = SVC()
clf.fit(train_features, labels)


answer = clf.predict(test_features)

submission_df = pd.DataFrame({'PassengerId' : test_df['PassengerId'],
                              'Survived'    : answer})

submission_csv = submission_df.to_csv(path_or_buf = 'submission.csv', index=False)
