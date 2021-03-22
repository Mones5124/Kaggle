#First Kaggle Data test, done mostly from other people's code using Student Performance

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from time import sleep
# https://www.kaggle.com/ammaraahmad/exploratory-data-analysis-using-python
#Importing dataset from kaggle which is "StudentsPerformance.csv"
filename = "StudentsPerformance.csv"
data=pd.read_csv("/Users/juanm./Documents/Coding/Data Science?/Kaggle/"+filename)

#print(data.head()) #prints the first part of the data
print(data.tail()) #prints the last part of the data
#these two allow you to know what the possible data are without having to look through everything

#print(data.shape) #describes the columns and rows in the data
#print(data.describe()) #basic statistics data about the data

print(data.columns) #It shows total columns in dataset 
print(data.nunique()) #Check unique values in all columns

#Check unique values in specific column
#print(data['parental level of education'].unique())

"""Cleaning the DATA"""
#Check if any column has null value
#print(data.isnull().sum()) 
#no null values

#drop unnecessary data using dataframe.drop['column title', axis = 1]
#Droping unnecessary Data
#student= data.drop(['race/ethnicity','parental level of education'],axis=1)
#print(student.head())

#adding precentages
data['Percentage'] = (data['math score']+data['reading score']+data['writing score'])/3

def Grade(Percentage):
    if (Percentage >= 95):return 'O'
    if (Percentage >= 81):return 'A'
    if (Percentage >= 71):return 'B'
    if (Percentage >= 61):return 'C'
    if (Percentage >= 51):return 'D'
    if (Percentage >= 41):return 'E'
    else: return 'F'
    
data["grade"] = data.apply(lambda x : Grade(x["Percentage"]), axis=1)
#loops through all of the values to apply the same function to it
print(data.head(10))
print(data.describe())

"""Data Visualization"""
#https://www.kaggle.com/bhartiprasad17/student-academic-performance-analysis
"""
#plot  for female to male pie chart
plt.figure(figsize=(14, 7))
labels=['Female', 'Male']
plt.pie(data['gender'].value_counts(),labels=labels,explode=[0.1,0.1],
        autopct='%1.2f%%',colors=['#E37383','#FFC0CB'], startangle=90)
plt.title('Gender')
plt.axis('equal')
plt.show()

#plot for score bar chart by gender
plt.figure(figsize=(10,5))
sns.set_context("talk",font_scale=1)
sns.set_palette("pastel")
ax = sns.countplot(y="grade", hue="gender", data=data, order=["O","A","B","C","D","E","F"])
ax.legend(loc='upper right',frameon=True)
plt.title('Gender vs Grades', fontsize=18, fontweight='bold')
ax.set(xlabel='COUNT',ylabel='GRADE')
plt.show()


#Correlation Heatmap
plt.figure(figsize=(8,8))
plt.title('Correlation Analysis',color='Red',fontsize=20,pad=40)
corr = data.corr()
mask = np.triu(np.ones_like(corr,dtype = bool))
sns.heatmap(data.corr(),mask=mask,annot=True,linewidths=.5);
plt.xticks(rotation=60)
plt.yticks(rotation = 60)
plt.show()

#Score Kde Plot
sns.set_context("paper",font_scale=1)
sns.kdeplot(data=data,shade = True)
plt.xlabel('Score')
plt.title('Score Kde Plot', fontsize=15, fontweight='bold')
plt.show()

#Versus table
sns.set_context("notebook")
sns.jointplot(data=data, x="math score", y="reading score", hue="gender", kind="kde")
plt.title('Reading and Mathematics score vs Gender', fontsize=15, fontweight='bold',y=1.3,loc="right")
plt.show()

#Percentage versus Test Perparation
sns.set_context("talk",font_scale=0.5)
sns.set_palette("Pastel2")
sns.kdeplot(data=data, x="Percentage", hue="test preparation course", multiple="stack")
plt.title('Percentage vs Test Preparation',fontsize=15, fontweight='bold')
plt.show()

#correlation table with box plot
sns.set_context("notebook")
sns.set_palette("pastel")
g = sns.JointGrid(data=data, x="Percentage", y="reading score")
g.plot(sns.regplot, sns.boxplot)
plt.title('Percentage and Reading score Relationship', fontsize=15, fontweight='bold',y=1.3,loc="right")
plt.show()

#shows a scatterplot
sns.set_palette("Pastel1")
g = sns.JointGrid(data=data, x="Percentage", y="math score", hue="test preparation course")
g.plot(sns.scatterplot, sns.histplot)
plt.title('Percentage and Mathematics score vs Test Preparation ', fontsize=15, fontweight='bold',y=1.3,loc="right")
plt.show()

#Percentage Distribution versus Gender
sns.set_context("notebook",font_scale=1)
sns.kdeplot(
   data=data, x="Percentage", hue="gender",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)
plt.title('Percentage Distribution w.r.t. Gender',fontsize=15, fontweight='bold')
plt.show()

#correlation between parent's education
sns.set_palette("Dark2")
sns.set_context("notebook",font_scale=1)
sns.kdeplot(
    data=data, x="Percentage", hue="parental level of education",
    cumulative=True, common_norm=False, common_grid=True,
)
plt.title('Percentage vs Parental Level Of Education',fontsize=15, fontweight='bold')
plt.show()

#Percentage Distrubition verus Race/Ethnicity
sns.set_palette("Set2")
(sns.FacetGrid(data,hue="race/ethnicity", height=5,xlim = (0,100)).map(sns.kdeplot, "Percentage").add_legend())
plt.title('Percentage Distribution w.r.t. Race/ethnicity',fontsize=15, fontweight='bold')
plt.show()
"""
"""Relationship Analysis"""
#use seaborn to try to find the relationship between variables

#sns.set()
#sns.pairplot(student)
#sleep(10)

#sns.relplot(x='math score',y='reading score',hue='gender',data=student)
#plots aren't actually working??
#why does Python 3.8 open without actually showing any graph and then close when the code finishes running

#try learning matplotlib and using it instead

plt.figure(figsize=(8,8))
plt.title('Correlation Analysis',color='Red',fontsize=20,pad=40)
corr = data.corr() #finds correlation relationship between the data
mask = np.triu(np.ones_like(corr, dtype = bool)) #only uses boolean data (since you can only find correlations between data)
sns.heatmap(corr,mask=mask,annot=True,linewidths=.5)
plt.xticks(rotation=60)
plt.yticks(rotation = 60)
plt.show()
