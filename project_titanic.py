# Import the relevant python libraries for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display 
#% matplotlib inline

#load the train dataset
train = pd.read_csv('train.csv')


#inspect the first few rows of the train dataset
display(train.head())


# set the index to passengerId
train = train.set_index('PassengerId')

#load the test dataset
test = pd.read_csv('test.csv')

display(test.head())


#by calling the shape attribute of the train dataset we can observe that there are 891 observations and 11 columns
#in the data set
train.shape


# Check out the data summary
# Age, Cabin and Embarked has missing data
train.head()


# identify datatypes of the 11 columns, add the stats to the datadict
datadict = pd.DataFrame(train.dtypes)
datadict

# identify missing values of the 11 columns,add the stats to the datadict
datadict['MissingVal'] = train.isnull().sum()
datadict


# Identify number of unique values, For object nunique will the number of levels
# Add the stats the data dict
datadict['NUnique']=train.nunique()
datadict


# Identify the count for each variable, add the stats to datadict
datadict['Count']=train.count()
datadict



# rename the 0 column
datadict = datadict.rename(columns={0:'DataType'})
datadict


# get discripte statistcs on "object" datatypes
train.describe(include=['object'])


# get discriptive statistcs on "number" datatypes
train.describe(include=['number'])


train.Survived.value_counts(normalize=True)

# Visualization on the basis of Bivariate Analysis
figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))
train.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
train.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
train.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
train.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
train.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
sns.boxplot(x="Survived", y="Age", data=train,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=train,ax=axesbi[1,2])
'''
Summary
1: We can clearly see that male survial rates is around 20% where as female survial rate is about 75% which suggests that gender has a strong relationship with the survival rates.
2: There is also a clear relationship between Pclass and the survival by referring to first plot below. Passengers on Pclass1 had a better survial rate of approx 60% whereas passengers on pclass3 had the worst survial rate of approx 22%
3: There is also a marginal relationship between the fare and survial rate.
4: I have quantified the above relationships further in the last statsical modelling section
'''




