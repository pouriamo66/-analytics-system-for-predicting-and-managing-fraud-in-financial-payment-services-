#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier 


# In[2]:


####import dataset
df=pd.read_csv("fraud.csv") 


# In[179]:


df.columns


# In[3]:


#####this one is same as sample 
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig',                         'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})


# In[4]:


df1=pd.read_csv("fraud.csv") 
df1=df1.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig',                         'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})


# In[5]:


df1['errorBalanceOrig']=round(df1['newBalanceOrig']+df1['amount']-df1['oldBalanceOrig'])
df1['errorBalanceDest']=round(df1['oldBalanceDest']+df1['amount']-df1['newBalanceDest'])


# In[ ]:





# In[22]:


for i in df.columns:
    print(i)
    print(np.unique(df[i]).shape)
    print('---------------------------------------------')


# In[23]:


df['isFraud'].value_counts()


# In[25]:


df1[df1['nameOrig']=='C2098525306'][['nameDest','amount','type','isFraud']]


# In[ ]:


C2098525306,
C400299098     3
C1902386530    3
C1065307291    3
C724452879     3
C545315117     3
C1832548028    3
C1976208114    3
C1462946854    3
C1999539787    3
C1784010646    3
C1530544995    3
C2051359467    3
C363736674     3
C1677795071    3


# In[16]:


np.sum(df1['nameOrig'].value_counts()>2)


# In[17]:


df1['nameDest'].value_counts()


# In[13]:


np.sum(df1['nameDest'].value_counts()>90)


# In[4]:


#df1['nameDest'].value_counts()
df1['type'].value_counts()


# In[ ]:





# In[6]:


######all data informations 
def datainfo(df):
    print("the first 5 row of dataframe is :")
    print(df.head())
    print('\n\n')
    print('##################################################################################################')
    print("the type of entitines  of dataframe are :")
    df.info()
    print('\n\n')
    print('##################################################################################################')
    print("the statistical information of dataframe are :")
    print('\n\n')
    print(df.describe())
    print('\n\n')
    print('##################################################################################################')
    print("the index  of dataframe are :")
    print('\n\n')
    print(df.index)
    print('\n\n')
    print('##################################################################################################')
    print("the column's names of dataframe are :")
    print('\n\n')
    print(df.columns)
    print('\n\n')
    print('##################################################################################################')
    print("the shape of dataframe is:")
    print('\n\n')
    print(df.shape)
    print('\n\n')
    print('##################################################################################################')
    print("Test if there any missing values in DataFrame:")
    print('\n\n')
    print(df.isnull().values.any())
    


# In[275]:


datainfo(df)


# In[173]:


print(df.type.value_counts())

f, ax = plt.subplots(1, 1, figsize=(8, 8))
df.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8))
plt.show()


# In[174]:


ax = df.groupby(['type', 'isFraud']).size().plot(kind='bar')
ax.set_title("# of transaction which are the actual fraud per transaction type")
ax.set_xlabel("(Type, isFraud)")
ax.set_ylabel("Count of transaction")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))


# In[4]:


print('\nAre there any merchants among originator accounts for CASH_IN transactions? {}'.format((df.loc[df.type == 'CASH_IN'].nameOrig.str.contains('M')).any())) 


# In[5]:


print('\nAre there any merchants among destination accounts for CASH_OUT transactions? {}'.format((df.loc[df.type == 'CASH_OUT'].nameDest.str.contains('M')).any())) # False


# In[6]:


print('\nAre there merchants among any originator accounts? {}'.format(      df.nameOrig.str.contains('M').any())) # False

print('\nAre there any transactions having merchants among destination accounts other than the PAYMENT type? {}'.format((df.loc[df.nameDest.str.contains('M')].type != 'PAYMENT').any())) # False


# In[ ]:


###############Exploratory Data Analysis


# In[7]:


def fraudulent(dataframe,column1,column2,column3,action1,action2,value1,value2):
    global Fraud_Transfer
    global Fraud_Cashout
    global Transfer
    global Flagged 
    global Not_Flagged
    global Not_Fraud
    print('\n The types of fraudulent transactions are ')
    print(dataframe.loc[dataframe[column1]== value1].type.drop_duplicates().values)
    print('\nThe type of transactions in which isFlaggedFraud is set:\{}'.format(list(dataframe.loc[dataframe[column3] == value1].type.drop_duplicates())))
    Fraud_Transfer = dataframe.loc[(dataframe[column1] == value1) & (dataframe[column2] == action1)]
    Fraud_Cashout = dataframe.loc[(dataframe[column1] == value1) & (dataframe[column2] == action2)]
    Transfer = dataframe.loc[dataframe[column2]== action1]
    Flagged = dataframe.loc[dataframe[column3] == value1]
    Not_Flagged = dataframe.loc[dataframe[column3] == value2]
    Not_Fraud = dataframe.loc[dataframe.isFraud == value2]
    print ('\n The number of fraudulent TRANSFERs = {}'.format(len(Fraud_Transfer))) 
    print ('\n The number of fraudulent CASH_OUTs = {}'.format(len(Fraud_Cashout))) 
    print('\n the Minimum amount transacted when isFlaggedFraud is:{}'.format(Flagged.amount.min()))
    print('\n the Maximum  amount transacted in a TRANSFER where isFlaggedFraud is :{}'.format(Transfer.loc[Transfer[column3] == 0].amount.max()))
    


# In[8]:


fraudulent(df,'isFraud','type','isFlaggedFraud','TRANSFER','CASH_OUT',1,0)


# In[6]:


fraudulent(dataset1,'isFraud','type','isFlaggedFraud','TRANSFER','CASH_OUT',1,0)


# In[9]:


def Destination():
    global Fraudulent_Dest
    Dest_Transfer=Fraud_Transfer.nameDest.isin(Fraud_Cashout.nameOrig).any()
    Fraudulent_Dest=Fraud_Transfer.loc[Fraud_Transfer.nameDest.isin(Not_Fraud.loc[Not_Fraud.type == 'CASH_OUT'].nameOrig.drop_duplicates())]
    IsOrgin_Fraud=Flagged.nameOrig.isin(pd.concat([Not_Flagged.nameOrig,Not_Flagged.nameDest])).any()
    IsDest_Orgin=Flagged.nameDest.isin(Not_Flagged.nameOrig).any()
    DestCount=sum(Flagged.nameDest.isin(Not_Flagged.nameDest))
    print('\nHave originators of transactions flagged as fraud transacted more than once?',IsOrgin_Fraud)
    print('\nHave destinations for transactions flagged as fraud initiated other transactions?',IsDest_Orgin)
    print('\nWithin fraudulent transactions, are there destinations for TRANSFERS that are also originators for CASH_OUTs?',Dest_Transfer)
    print('\nFraudulent TRANSFERs whose destination accounts are originators of genuine CASH_OUTs: \n\n',Fraudulent_Dest)
    print('\nHow many destination accounts of transactions flagged as fraud have been destination accounts more than once?:',DestCount)


# In[10]:


Destination()


# In[16]:


def tranStep(stepid):
    global step_cashout
    global step_transfer
    step_transfer=df[df.nameDest==stepid].step.values
    step_cashout=Not_Fraud.loc[(Not_Fraud.type == 'CASH_OUT') & (Not_Fraud.nameOrig ==stepid)].step.values
    print('Fraudulent TRANSFER',stepid,' occured at step:',step_transfer,'whereas genuine CASH_OUT from this account occured earlier at step :',step_cashout)


# In[40]:


a='C423543548'
tranStep(a)
    


# In[11]:


def FlaggedBalance():
    global F_balanced
    F_balanced=len(Transfer.loc[(Transfer.isFlaggedFraud == 0) & (Transfer.oldBalanceDest == 0) & (Transfer.newBalanceDest == 0)])
    print('\nThe number of TRANSFERs where isFlaggedFraud = 0, yet oldBalanceDest = 0 and\newBalanceDest = 0:',F_balanced)


# In[12]:


FlaggedBalance()


# In[13]:


def minmaxbalance():
    global minold
    global maxold
    global minnew
    global maxnew
    minold=Flagged.oldBalanceOrig.min()
    maxold=Flagged.oldBalanceOrig.max()
    minnew=Transfer.loc[(Transfer.isFlaggedFraud == 0)&(Transfer.oldBalanceOrig== Transfer.newBalanceOrig)].oldBalanceOrig.min()
    maxnew=Transfer.loc[(Transfer.isFlaggedFraud == 0) & (Transfer.oldBalanceOrig == Transfer.newBalanceOrig)].oldBalanceOrig.max()
    print('\nMinimum and Maximum of oldBalanceOrig for isFlaggedFraud = 1 TRANSFERs:  {}'.format([round(minold),round(maxold)]))
    print('\nMinimum and Maximum of oldBalanceOrig for isFlaggedFraud = 0 TRANSFERs where oldBalanceOrig =newBalanceOrig: {}'.format([round(minnew),round(maxnew)]))


# In[14]:


minmaxbalance()


# In[60]:


#######################Data cleaning###############################


# In[15]:


def cleaning(data,column1,column2,column3,column4,column5,column6,column7,column8,column9):
    global X
    global Y
    global Xfraud
    global XnonFraud
    global fractF
    global fractG
    randomState = 5
    np.random.seed(randomState)
    X = data.loc[(data[column1] == 'TRANSFER') | (data[column1] == 'CASH_OUT')]
    Y = X[column5]
    del X[column5]
    X = X.drop([column2, column3, column4], axis = 1)
    X.loc[X[column1] == 'TRANSFER', 'type'] = 0
    X.loc[X[column1] == 'CASH_OUT', 'type'] = 1
    X[column1] = X[column1].astype(int)
    ################## Imputation of Latent Missing Values
    Xfraud = X.loc[Y == 1]
    XnonFraud = X.loc[Y == 0]
    fractF=len(Xfraud.loc[(Xfraud[column6] == 0) & (Xfraud[column7] == 0) & (Xfraud.amount)])/(1.0 * len(Xfraud))
    fractG=len(XnonFraud.loc[(XnonFraud[column6] == 0) & (XnonFraud[column7] == 0) & (XnonFraud.amount)])/(1.0 * len(XnonFraud))
    X.loc[(X[column6] == 0) & (X[column7] == 0) & (X.amount != 0),[column6,column7]] = - 1
    X.loc[(X[column8] == 0) & (X[column9] == 0) & (X.amount != 0),[column8,column9]] = np.nan
    print('####################################################################################')
    print('\nThe fraction of fraudulent transactions with \'oldBalanceDest\' = \'newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: ',format(fractF))
    print('####################################################################################') 
    print('\nThe fraction of genuine transactions with \'oldBalanceDest\' = \newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: ',format(fractG))
    print('####################################################################################')
    
  


# In[16]:


cleaning(df,'type','nameOrig', 'nameDest', 'isFlaggedFraud','isFraud','oldBalanceDest','newBalanceDest','oldBalanceOrig', 'newBalanceOrig')


# In[18]:


data_new = df.copy()
dataset1 = data_new.copy()


# adding feature HourOfDay to Dataset1 
dataset1["Dayofweek"] = np.nan 
dataset1["HourOfDay"] = np.nan # initializing feature column
dataset1.HourOfDay = data_new.step % 24
dataset1["Dayofweek"] = np.nan # initializing feature column
dataset1.Dayofweek = data_new.step % 7

print("Head of dataset1: \n", pd.DataFrame.head(dataset1))


# In[40]:


Y.head()


# In[19]:


Y2=Y.copy()
X2=X.copy()
X2["HourOfDay"]=dataset1["HourOfDay"]
X2["Dayofweek"]=dataset1["Dayofweek"]


# In[20]:


def Fengineering(column1,column2,column3,column4,column5):
    global newcol1
    global newcol2
    newcol1= X2[column1] + X2[column2]- X2[column3]
    newcol2= X2[column4] + X2[column2 ] - X2[column5]
Fengineering('newBalanceOrig','amount','oldBalanceOrig','oldBalanceDest','newBalanceDest')
X2['errorBalanceOrig']=newcol1
X2['errorBalanceDest']=newcol2
X2.head()


# In[17]:


def Fengineering(column1,column2,column3,column4,column5):
    global newcol1
    global newcol2
    newcol1= X[column1] + X[column2]- X[column3]
    newcol2= X[column4] + X[column2 ] - X[column5]
Fengineering('newBalanceOrig','amount','oldBalanceOrig','oldBalanceDest','newBalanceDest')
X['errorBalanceOrig']=newcol1
X['errorBalanceDest']=newcol2
X.head()


# In[23]:


#######################Feature Engineering ###############################


# In[70]:


def Fengineering(column1,column2,column3,column4,column5):
    global newcol1
    global newcol2
    newcol1= X[column1] + X[column2]- X[column3]
    newcol2= X[column4] + X[column2 ] - X[column5]
Fengineering('newBalanceOrig','amount','oldBalanceOrig','oldBalanceDest','newBalanceDest')
X['errorBalanceOrig']=newcol1
X['errorBalanceDest']=newcol2
X.head()


# In[71]:


#3FINDING HOURS AND DAYS/copy the dataset to new dataset,then make the new dataset and cange type to new name
data_new = df.copy()
# initializing feature column
data_new["type1"]=np.nan 
# filling feature column
data_new.loc[df1.nameOrig.str.contains('C') & df1.nameDest.str.contains('C'),"type1"] = "CC" 
data_new.loc[df1.nameOrig.str.contains('C') & df1.nameDest.str.contains('M'),"type1"] = "CM"
data_new.loc[df1.nameOrig.str.contains('M') & df1.nameDest.str.contains('C'),"type1"] = "MC"
data_new.loc[df1.nameOrig.str.contains('M') & df1.nameDest.str.contains('M'),"type1"] = "MM"


# In[72]:


# Subsetting data into observations with fraud and valid transactions:
fraud = data_new[data_new["isFraud"] == 1]
valid = data_new[data_new["isFraud"] == 0]
fraud = fraud.drop('type1', 1)
valid = valid.drop('type1',1)
data_new = data_new.drop('type1',1)


# In[73]:


####assume that transaction only occur when transaction type is either CASH_OUT or TRANSFER.
valid = valid[(valid["type"] == "CASH_OUT")| (valid["type"] == "TRANSFER")]
data_new = data_new[(data_new["type"] == "CASH_OUT") | (data_new["type"] == "TRANSFER")]


# In[74]:


fraud = data_new[data_new["isFraud"] == 1]
valid = data_new[data_new["isFraud"] == 0]


# In[75]:


###omitting the nameOrig and nameDest columns from analysis.
names = ["nameOrig","nameDest"]
fraud = fraud.drop(names, 1)
valid = valid.drop(names,1)
data_new = data_new.drop(names,1)


# In[76]:


#######omitting the isFlaggedFraud column from the analysis
fraud = df1[df1["isFraud"] == 1]
valid = df1[df1["isFraud"] == 0]
fraud = fraud.drop("isFlaggedFraud",1)
valid = valid.drop("isFlaggedFraud",1)


# In[56]:


###defining function to seee number of valid/fraud  transaction over time 
def fraudhist(X) :  
    bins = 60
    valid.hist(column="step",color="blue",bins=bins)
    plt.xlabel("1 hour time step")
    plt.ylabel("# of transactions")
    plt.title("# of valid transactions over time")
    X.hist(column ="step",color="orange",bins=bins)
    plt.xlabel("1 hour time step")
    plt.ylabel("# of transactions")
    plt.title("# of fraud transactions over time")
    plt.tight_layout()
    plt.show()


# In[57]:


fraudhist(fraud)


# In[160]:


# getting hours and days of the week
def DayHour(X,Y):
    global fraud_days
    global fraud_hours
    global valid_days
    global valid_hours
    num_days = 7
    num_hours = 24
    fraud_days = X.step % num_days
    fraud_hours = X.step % num_hours
    valid_days = Y.step % num_days
    valid_hours = Y.step % num_hours
    # plotting scatterplot of the days of the week, identifying the fraudulent transactions (red) from the valid transactions (yellow) 
    plt.subplot(2, 2, 1)
    fraud_days.hist(bins=num_days,color="red")
    plt.title('Fraud transactions by Day')
    plt.xlabel('Day of the Week')
    plt.ylabel("# of transactions")
    plt.subplot(2, 2, 2)
    valid_days.hist(bins=num_days,color="yellow")
    plt.title('Valid transactions by Day')
    plt.xlabel('Day of the Week')
    plt.ylabel("# of transactions")
    plt.subplot(2, 2, 3)
    fraud_hours.hist(bins=num_hours, color="blue")
    plt.title('Fraud transactions by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel("# of transactions")
    plt.subplot(2, 2, 4)
    valid_hours.hist(bins=num_hours, color="magenta")
    plt.title('Valid transactions by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel("# of transactions")
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.show()


# In[161]:


DayHour(fraud,valid)


# In[77]:


dataset1 = data_new.copy()


# adding feature HourOfDay to Dataset1 
dataset1["Dayofweek"] = np.nan 
dataset1["HourOfDay"] = np.nan # initializing feature column
dataset1.HourOfDay = data_new.step % 24
dataset1["Dayofweek"] = np.nan # initializing feature column
dataset1.Dayofweek = data_new.step % 7

print("Head of dataset1: \n", pd.DataFrame.head(dataset1))


# In[51]:


dataset1['nameOrig'].value_counts().head(15)
item=['C2098525306','C400299098','C1902386530','C1065307291','C724452879','C545315117','C1832548028','C1976208114','C1462946854','C1999539787','C1784010646','C1530544995' ,'C2051359467','C363736674','C1677795071']  
   


# In[71]:


df3=[]
for item_v  in item[0:14]:
    print('############################')
    print('the name of orgin is',item_v)
    print('############################')
    print(dataset1[dataset1['nameOrig']==item_v][['nameDest','Dayofweek','HourOfDay','amount','type','isFraud']])
    print('---------------------------------------------------------------------------------------------------------')
    df3.append(item_v)
    df3.append(dataset1[dataset1['nameOrig']==item_v][['nameDest','Dayofweek','HourOfDay','amount','type','isFraud']])


# In[73]:


dataset1['nameDest'].value_counts().head(15)  


# In[82]:


item1=['C1286084959','C985934102','C665576141','C2083562754','C248609774','C1590550415','C1789550256','C451111351','C1360767589','C1023714065','C97730845','C977993101' ,'C392292416','C1899073220 ','C306206744']


# In[99]:


df5=[]
for item_v1  in item1[0:14]:
    print('####################################################################################################################')
    print('the name of Dest is',item_v1)
    print('')
    print('---------------------------------------------------------------------------------------------------------')
    print(dataset1[dataset1['nameDest']==item_v1][['nameOrig','Dayofweek','HourOfDay','amount','type','isFraud']])
    print('---------------------------------------------------------------------------------------------------------')
    print('the avrage of amount is',np.mean(dataset1[dataset1['nameDest']==item_v1]['amount']))
    print('the minimum of amount is',min(dataset1[dataset1['nameDest']==item_v1]['amount']))
    print('the Maximum of amount is',max(dataset1[dataset1['nameDest']==item_v1]['amount']))
    print('the standard deviation of amount is',np.std(dataset1[dataset1['nameDest']==item_v1]['amount']))
    df5.append(item_v1)
    df5.append(dataset1[dataset1['nameDest']==item_v1][['nameOrig','Dayofweek','HourOfDay','amount','type','isFraud']])


# In[45]:


#######################Data visualization###############################


# In[58]:


limit = len(X)

def plotStrip(x, y, hue, figsize = (14, 9)):
    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y,hue = hue, jitter = 0.4, marker = '.',size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1),                loc=2, borderaxespad=0, fontsize = 16);
    return ax


# In[62]:


ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size = 6)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent transactions over time', size = 10);

fig,ax=plt.subplot(2, 2, 1)
ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size = 6)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent transactions over time', size = 10)
fig,ax=plt.subplot(2, 2, 2)
ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('amount', size = 16)
ax.set_title('Same-signed fingerprints of genuine and fraudulent transactions over amount', size = 18);


# In[52]:


limit = len(X)
ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('amount', size = 16)
ax.set_title('Same-signed fingerprints of genuine and fraudulent transactions over amount', size = 18);


# In[64]:


limit = len(X)
ax = plotStrip(Y[:limit], - X.errorBalanceDest[:limit], X.type[:limit],               figsize = (14, 9))
ax.set_ylabel('- errorBalanceDest', size = 16)
ax.set_title('Opposite polarity fingerprints over the error in destination account balances', size = 18);


# In[71]:


def plot3d(df1,df2,x,y,z,zOffset,limit):
    sns.reset_orig() # prevent seaborn from over-riding mplot3d defaults
    fig = plt.figure(figsize = (10, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df1.loc[df2 == 0, x][:limit], df1.loc[df2 == 0, y][:limit], -np.log10(df1.loc[df2== 0, z][:limit] + zOffset), c = 'b', marker = '.', s = 1, label = 'genuine')
    ax.scatter(df1.loc[df2== 1, x][:limit], df1.loc[df2 == 1, y][:limit],  -np.log10(df1.loc[df2 == 1, z][:limit] + zOffset), c = 'y', marker = '.', s = 1, label = 'fraudulent')
    ax.set_xlabel(x, size = 16); 
    ax.set_ylabel(y + ' [hour]', size = 16); 
    ax.set_zlabel('- log$_{10}$ (' + z + ')', size = 16)
    ax.set_title('Error-based features separate out genuine and fraudulent transactions', size = 20)
    plt.axis('tight')
    ax.grid(1)
    noFraudMarker = mlines.Line2D([], [], linewidth = 0, color='b', marker='.',markersize = 10, label='genuine')
    fraudMarker = mlines.Line2D([], [], linewidth = 0, color='y', marker='.',markersize = 10, label='fraudulent')
    plt.legend(handles = [noFraudMarker, fraudMarker],bbox_to_anchor = (1.20, 0.38 ), frameon = False, prop={'size': 16});


# In[72]:


plot3d(X,Y,'errorBalanceDest','step','errorBalanceOrig',0.02,len(X))


# In[18]:


def updateFraud(df1,df2):
    global x_fraud
    global x_nonfraud
    global corre_non_fraud
    global indices 
    global mask
    global corre_fraud
    x_fraud = df1.loc[df2== 1] 
    x_nonfraud = df1.loc[df2 == 0]
    corre_non_fraud = x_nonfraud.loc[:, df1.columns != 'step'].corr()
    corre_fraud = x_fraud.loc[:, df1.columns != 'step'].corr()
    mask = np.zeros_like( corre_non_fraud)
    indices = np.triu_indices_from( corre_non_fraud)
    mask[indices] = True
    Skew_fraud=len(x_fraud) / float(len(df1))
    print('####################################################################################')
    print('print head of x_fraud with cleaned data: ',x_nonfraud.head())
    print('####################################################################################')
    print('show the correlation heatmap for x_fraud with cleaned data\n ')
    print('print head of x_fraud with cleaned data: ',x_fraud.head())
    print('####################################################################################\n')
    print('show the Detect Fraud in Skewed Data: ',Skew_fraud)
   
    
   
    
    
    


# In[19]:


updateFraud(X,Y)


# In[213]:


print('show the correlation heatmap for Non-fraud with cleaned data\n ')
corre_non_fraud.style.background_gradient(cmap='coolwarm')


# In[95]:


corre_Non_Fraud.style.background_gradient(cmap='coolwarm')


# In[163]:


def plotfraud(df1):
       grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
       f,(ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws,figsize = (14, 9))
       cmap =sns.cubehelix_palette(8)
       ax1 =sns.heatmap(corre_non_fraud, ax = ax1, vmin = -1, vmax = 1,cmap = cmap, square = False, linewidths = 0.5, mask = mask, cbar = False)
       ax1.set_xticklabels(ax1.get_xticklabels(), size = 16); 
       ax1.set_yticklabels(ax1.get_yticklabels(), size = 16); 
       ax1.set_title('Genuine \n transactions', size = 20)
       ax2 = sns.heatmap(corre_fraud, vmin = -1, vmax = 1, cmap = cmap, ax = ax2, square = False, linewidths = 0.5, mask = mask, yticklabels = False,cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', 'ticks': [-1, -0.5, 0, 0.5, 1]})
       ax2.set_xticklabels(ax2.get_xticklabels(), size = 16); 
       ax2.set_title('Fraudulent \n transactions', size = 20);
       cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 14);


# In[148]:


plotfraud(X)


# In[176]:


dataset1.head()


# In[81]:


def updateFraud1(df1,df2):
    global x_fraud1
    global x_nonfraud1
    global corre_non_fraud1
    global indices1 
    global mask1
    global corre_fraud1
   
    x_fraud1 = df1.loc[df2== 1] 
    x_nonfraud1 = df1.loc[df2 == 0]
    corre_non_fraud1 = x_nonfraud1.loc[:, df1.columns != 'step'].corr()
    
    corre_fraud1 = x_fraud1.loc[:, df1.columns != 'step'].corr()
    mask1 = np.zeros_like( corre_non_fraud1)
    indices1 = np.triu_indices_from( corre_non_fraud1)
    mask1[indices1] = True
    Skew_fraud1=len(x_fraud1) / float(len(df1))
    print('####################################################################################')
    print('print head of x_fraud with cleaned data: ',x_nonfraud1.head())
    print('####################################################################################')
    print('show the correlation heatmap for x_fraud with cleaned data\n ')
    print('print head of x_fraud with cleaned data: ',x_fraud1.head())
    print('####################################################################################\n')
    print('show the Detect Fraud in Skewed Data: ',Skew_fraud1)
   


# In[ ]:


X1 = dataset1.loc[(dataset1['type'] == 'TRANSFER') | (dataset1['type'] == 'CASH_OUT')]
Y1 = X1['isFraud']
del X1['isFraud']
X1 = X1.drop(['isFlaggedFraud'], axis = 1)
X1.loc[X1['type'] == 'TRANSFER', 'type'] = 0
X1.loc[X1['type'] == 'CASH_OUT', 'type'] = 1
X1['type'] = X1['type'].astype(int)


# In[21]:


X = df.loc[(df['type'] == 'TRANSFER') | (df['type'] == 'CASH_OUT')]
Y = X['isFraud']
del X['isFraud']
X = X.drop(['isFlaggedFraud'], axis = 1)
X.loc[X['type'] == 'TRANSFER', 'type'] = 0
X.loc[X['type'] == 'CASH_OUT', 'type'] = 1
X['type'] = X['type'].astype(int)


# In[83]:


updateFraud1(X1,Y1)


# In[106]:


def plotfraud1(df1):
       grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
       f,(ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws,figsize = (14, 9))
       cmap =sns.cubehelix_palette(8)
       ax1 =sns.heatmap(corre_non_fraud1, ax = ax1, vmin = -1, vmax = 1,cmap = cmap, square = False, linewidths = 0.5, mask = mask1, cbar = False)
       ax1.set_xticklabels(ax1.get_xticklabels(), size = 16); 
       ax1.set_yticklabels(ax1.get_yticklabels(), size = 16); 
       ax1.set_title('Genuine \n transactions', size = 20)
       ax2 = sns.heatmap(corre_fraud1, vmin = -1, vmax = 1, cmap = cmap, ax = ax2, square = False, linewidths = 0.5, mask = mask1, yticklabels = False,cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', 'ticks': [-1, -0.5, 0, 0.5, 1]})
       ax2.set_xticklabels(ax2.get_xticklabels(), size = 16); 
       ax2.set_title('Fraudulent \n transactions', size = 20);
       cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 14);


# In[239]:


plotfraud1(X1)


# In[84]:


X.fillna(X.mean(), inplace=True)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2,                                                 random_state = 42)


# In[37]:


# Create the parameter grid: gbm_param_grid
import xgboost as xgb
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators':[50, 100, 150, 200],
    'max_depth': [5,10,25,35,40],
    'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBClassifier(objective="reg:logistic")

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring="neg_log_loss", cv=5, verbose=1)
grid_mse.fit(X,Y)
print("Best: %f using %s" % (grid_mse.best_score_, grid_mse.best_params_))


# In[100]:


import xgboost as xgb
clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }

numFolds = 5
kfold_5 = cross_validation.KFold(n = len(X2), shuffle = True, n_folds = numFolds)

clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = kfold_5,  
                         n_iter = 5, # you want 5 here not 25 if I understand you correctly 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)


# In[86]:


def xgbClassifier (df1, df2,a,b,n):
    global Xgb
    trainX, testX, trainY, testY = train_test_split(df1, df2, test_size = a,random_state = 42)
    weights = (df2 == 0).sum() / (1.0 * (df2 == 1).sum())
    Xgb= XGBClassifier(max_depth =b, scale_pos_weight = weights,n_jobs =n)
    probabilities = Xgb.fit(trainX, trainY).predict_proba(testX)
    print('AUPRC = {}'.format(average_precision_score(testY,probabilities[:, 1])))
   
    
    


# In[113]:


xgbClassifier(X,Y,0.2,3,4)


# In[137]:


xgbClassifier(X2,Y2,0.2,3,4)


# In[138]:


fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = ["r", "g", "b", "peachpuff", "orange","gray","yellow","m","c","black","gray"]

ax = plot_importance(Xgb, height = 1, color = colours, grid = False,                      show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);


# In[59]:


# Create the training and test sets
X_train,X_test,y_train,y_test= train_test_split(X,Y, test_size=0.3, random_state=123)
# Instantiate the XGBClassifier: xg_cl
gbm1 = XGBClassifier(objective="reg:logistic",colsample_bytree=0.7, max_depth=5,learning_rate=0.1,n_estimators=500)
# Fit the classifier to the training set
gbm1.fit(X_train,y_train)
# Predict the labels of the test set: preds
preds = gbm1.predict(X_test)
preds1 = gbm1.predict(X_train)
# Compute the accuracy: accuracy
accuracy_test = float(np.sum(preds==y_test))/y_test.shape[0]
accuracy_train = float(np.sum(preds1==y_train))/y_train.shape[0]
print("accuracy of test dataset: %f" % (accuracy_test))
print("accuracy of train dataset: %f" % (accuracy_train))


# In[63]:


cv_scores2 = cross_val_score(gbm1,X,Y,cv=3)
# Print the 5-fold cross-validation scores
print(cv_scores2)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores2)))


# In[66]:


# fit the model with data
gbm1.fit(X_train,y_train)
y_pred3=gbm1.predict(X_test)
y_pred_prob1 = gbm1.predict_proba(X_test)[:,1]
y_pred_prob1 = gbm1.predict_proba(X_test)[:,1]


# In[69]:


def EvaluationMetrics1(K,Z,z_test,z_train,z_pred):
   print('####################################################################################')
   # Model Accuracy, how often is the classifier correct?
   print("Accuracy:",metrics.accuracy_score(z_test, z_pred))
   print('####################################################################################')
   print("Precision:",metrics.precision_score(z_test, z_pred))
   print('####################################################################################')
   print("Recall:",metrics.recall_score(z_test, z_pred))
   print('#####################confusion_matrix########################################')
   print(confusion_matrix(z_test, z_pred))
   print('#####################classification_report########################################')
   print(classification_report(z_test, z_pred))
   


# In[70]:


EvaluationMetrics1(X,Y,y_test,y_train,y_pred3)  


# In[64]:


ROC1(gbm1)


# In[ ]:


#################Bias-variance tradeoff


# In[140]:


def bias(K,Z,K_train,Z_train,a,b):
   global trainScoresMean
   global trainScoresStd 
   global crossValScoresMean
   global crossValScoresStd
   global trainSizes
   global trainScoresStd 
   global crossValScores
   global trainScores 
   weights = (Z == 0).sum() / (1.0 * (Z == 1).sum())
   trainSizes, trainScores, crossValScores = learning_curve(XGBClassifier(max_depth =a, scale_pos_weight = weights, n_jobs = b), K_train,Z_train, scoring = 'average_precision')
   trainScoresMean = np.mean(trainScores, axis=1)
   trainScoresStd = np.std(trainScores, axis=1)
   crossValScoresMean = np.mean(crossValScores, axis=1)
   crossValScoresStd = np.std(crossValScores, axis=1)
   print('#####################trainScoresMean##################################')
   print(trainScoresMean)
   print('#####################trainScoresStd##################################')
   print(trainScoresStd)
   print('#####################crossValScoresMean##################################')
   print(crossValScoresMean)
   print('#####################crossValScoresStd##################################')
   print(crossValScoresStd)


# In[24]:


bias(X,Y,trainX,trainY,3,4)


# In[144]:


bias(X2,Y2,trainX2,trainY2,3,4)


# In[89]:


def biasGragh() :
   colours = plt.cm.tab10(np.linspace(0, 1, 9))
   fig = plt.figure(figsize = (14, 9))
   plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
   trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
   plt.fill_between(trainSizes, crossValScoresMean - crossValScoresStd,
   crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])
   plt.plot(trainSizes, trainScores.mean(axis = 1), 'o-', label = 'train', color = colours[0])
   plt.plot(trainSizes, crossValScores.mean(axis = 1), 'o-', label = 'cross-val',color = colours[1])
   ax = plt.gca()
   for axis in ['top','bottom','left','right']:
     ax.spines[axis].set_linewidth(2)
   handles, labels = ax.get_legend_handles_labels()
   plt.legend(handles, ['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15),loc=2, borderaxespad=0, fontsize = 16);
   plt.xlabel('training set size', size = 16); 
   plt.ylabel('AUPRC', size = 16)
   plt.title('Learning curves indicate slightly underfit model', size = 20);


# In[90]:


biasGragh()


# In[145]:


def biasGragh1() :
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    fig = plt.figure(figsize = (14, 9))
    plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
    trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
    plt.fill_between(trainSizes, crossValScoresMean - crossValScoresStd,
    crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])
    plt.plot(trainSizes, trainScores.mean(axis = 1), 'o-', label = 'train', color = colours[0])
    plt.plot(trainSizes, crossValScores.mean(axis = 1), 'o-', label = 'cross-val',color = colours[1])
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(2)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, ['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15),loc=2, borderaxespad=0, fontsize = 16);
    plt.xlabel('training set size', size = 16); 
    plt.ylabel('AUPRC', size = 16)
    plt.title('Learning curves indicate slightly underfit model', size = 20);


# In[147]:


biasGragh1()


# In[ ]:


###########################logistic


# In[27]:


X.drop(['nameOrig','nameDest'], axis = 1, inplace = True)


# In[75]:


# ####Hyperparameters
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
logreg = LogisticRegression()


# In[76]:


logreg_cv=GridSearchCV(logreg,grid,cv=3)
logreg_cv.fit(X,Y)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[49]:


def EvaluationMetrics(K,Z,z_test,z_train,z_pred):
   print('####################################################################################')
   # Model Accuracy, how often is the classifier correct?
   print("Accuracy:",metrics.accuracy_score(z_test, z_pred))
   print('####################################################################################')
   print("Precision:",metrics.precision_score(z_test, z_pred))
   print('####################################################################################')
   print("Recall:",metrics.recall_score(z_test, z_pred))
   print('#####################confusion_matrix########################################')
   print(confusion_matrix(z_test, z_pred))
   print('#####################classification_report########################################')
   print(classification_report(z_test, z_pred))
   print('#####################Cross validation kfold=5 ########################################')
   print(cross_val_score(logreg,K,Z,cv=5) )


# In[149]:


def ROC(K,Z,z_test, z_pred_prob):   
    fpr, tpr, thresholds = roc_curve(z_test, z_pred_prob)
    # create plot
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    _ = plt.xlabel('False Positive Rate')
    _ = plt.ylabel('True Positive Rate')
    _ = plt.title('ROC Curve')
    _ = plt.xlim([-0.02, 1])
    _ = plt.ylim([0, 1.02])
    _ = plt.legend(loc="lower right")


# In[51]:


def ROC1(model):
    model.fit(X_train,y_train)
    y_probas = model.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas)


# In[31]:


logreg.fit(X_train,y_train)


# In[32]:


print(" training accuracy:", logreg.score(X_train, y_train))
print(" test accuracy    :", logreg.score(X_test, y_test))


# In[34]:


# Compute 10-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(logreg,X,Y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[52]:


ROC1(logreg)


# In[62]:


# fit the model with data
logreg.fit(X_train,y_train)
y_pred1=logreg.predict(X_test)
y_pred_prob1 = gnb.predict_proba(X_test)[:,1]
y_pred_prob1 = gnb.predict_proba(X_test)[:,1]


# In[69]:


EvaluationMetrics(X,Y,y1_test,y1_train,y_pred1)  


# In[93]:


ROC(X,Y,y1_test, y_pred_prob1)


# In[ ]:


###################Navie


# In[ ]:


grid_params = {
  'mnb__alpha': np.linspace(0.5, 1.5, 6),
  'mnb__fit_prior': [True, False],
  'tfidf_pip__tfidf_vectorizer__max_df': np.linspace(0.1, 1, 10),
  'tfidf_pip__tfidf_vectorizer__binary': [True, False],
  'tfidf_pip__tfidf_vectorizer__norm': [None, 'l1', 'l2'], 
}
clf = GridSearchCV(gnb, grid_params)
clf.fit(X_train, y_train)
print("Best Score: ", clf.best_score_)
print("Best Params: ", clf.best_params_)


# In[77]:


####Hyperparameters
tuned_parameters = { 'tfidf__use_idf': (True, False),'tfidf__norm': ('l1', 'l2'),'alpha': [1, 1e-1, 1e-2]}
gnb_cv=GridSearchCV(gnb,tuned_parameters,cv=3)
gnb_cv.fit(X,Y)
print("tuned hpyerparameters :(best parameters) ",gnb_cv.best_params_)
print("accuracy :",gnb_cv.best_score_)


# In[38]:


gnb = GaussianNB()
gnb.fit(X_train,y_train)
print(" training accuracy:", gnb.score(X_train, y_train))
print(" test accuracy    :", gnb.score(X_test, y_test))


# In[40]:



# Compute 10-fold cross-validation scores: cv_scores
cv_scores1 = cross_val_score(gnb,X,Y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores1)))


# In[39]:


ROC1(gnb)


# In[80]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=109) # 70% training and 30% test
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
y_pred_prob = gnb.predict_proba(X_test)[:,1]


# In[94]:


EvaluationMetrics(X,Y,y_test,y_train,y_pred)  


# In[108]:


ROC(X,Y,y_test, y_pred_prob)


# In[ ]:


######################RandomForest


# In[79]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


###Fit the grid search to the data
grid_search.fit(X,Y)
grid_search.best_params_
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))


# In[43]:


Rf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy', 

            max_depth=10, max_features='auto', max_leaf_nodes=4, 

            min_impurity_decrease=0.0, min_impurity_split=None, 

            min_samples_leaf=1, min_samples_split=4, 

            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1, 

            oob_score=False, random_state=10, verbose=1, 

            warm_start=False) 


# In[45]:


rf=RandomForestClassifier()
rf.fit(X_train,y_train)


# In[46]:


print(" training accuracy:", rf.score(X_train, y_train))
print(" test accuracy    :", rf.score(X_test, y_test))


# In[47]:


# Compute 10-fold cross-validation scores: cv_scores
cv_scores1 = cross_val_score(rf,X,Y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores1)))


# In[50]:


y2_pred = rf.predict(X_test)
y2_pred_prob = rf.predict_proba(X_test)[:,1]
EvaluationMetrics(X,Y,y_test,y_train,y2_pred)  


# In[53]:


ROC1(rf)


# In[72]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
dot_data = StringIO()  
export_graphviz(estimator, out_file=dot_data,feature_names=X.columns,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 

