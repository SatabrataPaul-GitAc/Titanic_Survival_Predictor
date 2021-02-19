#!/usr/bin/python3
print("content-type:text/html")
print()


#The below modules will be required for model training as well as data conversion and manipulation . 
import cgi
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


#The data that is submitted by the user , is collected and stored in separate vaiables .
NAME = (cgi.FieldStorage()).getvalue("nm")
GENDER = (cgi.FieldStorage()).getvalue("gender")
AGE = (cgi.FieldStorage()).getvalue("age")
PCLASS = (cgi.FieldStorage()).getvalue("pclass")
PARCH = (cgi.FieldStorage()).getvalue("parch")
EMBARK = (cgi.FieldStorage()).getvalue("embark")


#A list that will be used to store the user data as logical numeric values , as a sequence .
#In Machine Learning , only numeric data makes sense which is essential for model training , thus these below functions that are declared help in creating that logical numeric sequence and store in the list.

user= list()

#Function 1
def gender_convert(g):

   if g=="Male":
      user.append(1)
   else:
      user.append(0)

   
#Function 2
def pclass_convert(pcl):
   
   nl=list(np.zeros(2,dtype=int))

   if pcl == '2':
      nl[0]=1
   elif pcl == '3':
      nl[1]=1

   user.extend(nl)


#Function 3
def embark_convert(emb):
   
   nl=list(np.zeros(2,dtype=int))

   if emb=='Q':
      nl[0]=1
   elif emb=='S':
      nl[1]=1

   user.extend(nl)


#Function 4
def parch_convert(prc):
   
   nl=list(np.zeros(3,dtype=int))
   
   if prc=='1':
      nl[0]=1
   elif prc=='2':
      nl[1]=1
   elif prc=='3':
      nl[2]=1

   user.extend(nl)
      


#Calling all the above declared functions to implement them inorder to store, logically numeric values . 
gender_convert(GENDER)
pclass_convert(PCLASS)
embark_convert(EMBARK)
parch_convert(PARCH)
user.append(int(AGE))

final_input=[user]



#The code for feature selection and data manipulation
dataset=pd.read_csv("titanic.csv")
y=dataset['Survived']

gender=pd.get_dummies(dataset['Sex'],drop_first=True)

pcl=pd.get_dummies(dataset['Pclass'],drop_first=True)
pcl.rename(columns={2:"class 2",3:"class 3"},inplace=True)

emb=pd.get_dummies(dataset['Embarked'],drop_first=True)

parch=pd.get_dummies(dataset['Parch'])
parch.drop([0,4,5,6],axis=1,inplace=True)
parch.rename(columns={1:"ratio 1",2:"ratio 2",3:"ratio 3"},inplace=True)

#The below function is created to fill out missing values in the age field of the dataset . To determine what value to be filled , the help of data visualization libraries like seaborn and matplotlib are taken.
def fill(col):
   
   pcl=col[0]
   age=col[1]

   if pd.isnull(age):
      if pcl==1:
         return 38
      elif pcl==2:
         return 30
      elif pcl==3:
         return 25

   else:
      return age

age= dataset[['Pclass','Age']].apply(fill,axis=1)
age=pd.DataFrame(age)
age.rename(columns={0:"age"},inplace=True)


final_dataset= pd.concat([gender,pcl,emb,parch,age],axis=1)

#Training the model for binary classification using logistic regression . 
model = LogisticRegression()
model.fit(final_dataset,y)

pred = model.predict(final_input)

#Condition for displaying the survival based on prediction
if pred==0:
  print("""<h1 style="text-align:center;color:red";> NOT SURVIVE </h1>""")
else:
  print("""<h1 style="text-align:center;color:green";> SURVIVE </h1>""")






