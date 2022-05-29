import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 

k=3

df=pd.read_csv(r'C:\other\FSM\level2_ml\train.csv')

df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Embarked']=df['Embarked'].map({'S':1,'C':2,'Q':3})
#print(np.sum(df.isna()))
df=df.drop(['Cabin'],axis=1)
df['Age']=df['Age'].fillna(method='backfill')
df['Embarked']=df['Embarked'].fillna(method='ffill')
#print(np.sum(df.isna()))
#print(df.columns)

y=df[['Survived']].values
x=df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values



def dif(x_t,k):
    diff=[]
    
    for i in range(len(x)):
        sum=0
        
        for j in range(7):
            a=((x[i][j]-x_t[j])**2)
            sum+=a
        diff.append(sum**0.5)
        
    di=np.array(diff)
    dif=pd.Series(di)
    
    df1=df.assign(Diff=dif)
    y1=df1[['Survived','Diff']]
    min2=y1.sort_values('Diff')
    #print(y1)   
    select=min2[:k]
    a=list(select['Survived'])
    #print(min2)
    a_0=a.count(0)
    a_1=a.count(1)
    if a_0 > a_1:
        return 0
    else:
        return 1    



test_df=pd.read_csv(r"C:\other\FSM\level2_ml\test.csv")
test_df['Sex']=test_df['Sex'].map({'male':0,'female':1})
test_df['Embarked']=test_df['Embarked'].map({'S':1,'C':2,'Q':3})
#print(np.sum(df.isna()))
test_df=test_df.drop(['Cabin'],axis=1)
test_df['Age']=test_df['Age'].fillna(method='backfill')
test_df['Embarked']=test_df['Embarked'].fillna(method='ffill')

test_x=test_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
test_y=[]
#print(test_x)

def test(x_t,k):
    
    for i in range(len(test_x)):
        t_y=dif(x_t[i],k)
        
        test_y.append(t_y)
    print(test_y)


test(test_x,k)

t_1=test_y.count(1)
t_0=test_y.count(0)
y=[]
y.append(t_1)
y.append(t_0)


plt.pie(y,labels=['Not Survived(0)','Survived(1)'],autopct='%1.2f%%')
plt.show()




