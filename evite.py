#!/usr/bin/env python
# coding: utf-8
get_ipython().run_line_magic('pip', 'install mock --user')
get_ipython().run_line_magic('pip', 'install pmdarima --user')

# # Predicting Adding Donation Feature
# ## Import Libraries
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# load events data
events = pd.read_csv('evite-reports/events.csv')
events.head()

# load donations data
donations = pd.read_csv('evite-reports/donations.csv')
donations.head()

# change column name from External ID to Event ID
donations = donations.rename(columns={"External ID": "Event ID"})

# read in organizations data
org = pd.read_csv('evite-reports/organizations.csv')
org.head()

# ## Data Types of three tables
events.dtypes
donations.dtypes
org.dtypes


# ## Number of rows and columns for each table
events.shape
donations.shape
org.shape

# ## How many NA values are in each column for Event Table?
events.isnull().sum()

# ## What are the distributions for Beneficiary Type Column from Events Table?
events['Beneficiary Type'].value_counts()
# ## We need to exclude data with NA in Event Dates and filter for date range from 2017 to 2020
events = events.dropna(subset=['Event Date'])
events['Event Date'] = pd.to_datetime(events['Event Date'], errors = 'coerce', utc=True)
nts = events[events['Event Date'] < '2020-04-01']
events = events[events['Event Date'] > '2017-01-01']
print('most recent: ', max(events['Event Date']))
print('oldest: ',  min(events['Event Date']))

# ## Make a new column for labels: Added Donation - 1, Didn't add Donation - 0
# Data for Events (Logistic Regression)
df_events = events
df_events['Donation'] = np.where(df_events['Beneficiary Type'].isna(), 0 , 1)
df_events.dtypes

# ## Drop columns not needed for this modeling and check if there are any more NA values
df_events = df_events.dropna(subset=['Category','Zip code'])
df_events = df_events.drop(columns = ['Beneficiary Type','Internal ID','Host email','Total Raised','Total donors','Goal','Nonprofit ID'])
df_events['Quarter'] = df_events['Event Date'].dt.quarter
df_events.isnull().sum()

# ## Mapping files are needed for grouping categorical variables
# Other Data
event_mapping = pd.read_csv('evite-reports/event_mapping.csv')
org_mapping = pd.read_csv('evite-reports/org_mapping.csv')
state_mapping = pd.read_csv('evite-reports/state_mapping.csv')
state_wages = pd.read_csv('evite-reports/state_wages.csv')
zipcode_mapping = pd.read_csv('evite-reports/zip_code_database.csv')

# ## filter out for US only events
zipcode_mapping = zipcode_mapping[zipcode_mapping['country'] == 'US']
zipcode_mapping['country'].value_counts()
zipcode_mapping = zipcode_mapping[['zip','state']]

# ## These are available columns from mapping files
print('event_mapping:')
for col in event_mapping.columns: 
    print(col)
print("")
print('state_wages:')
for col in state_wages.columns: 
    print(col)
print("")
print('zipcode_mapping:')
for col in zipcode_mapping.columns: 
    print(col)
print("")
print('state_mapping:')
for col in state_mapping.columns: 
    print(col)

event_mapping.dtypes

df_events['Zip code'] = df_events['Zip code'].astype(str)
df_events.dtypes

zipcode_mapping['zip'] = zipcode_mapping['zip'].astype(str)
zipcode_mapping.dtypes

# ## Mappings for Category to Grouped Category, Zipcode to State, State to Region
df_events = pd.merge(df_events, event_mapping, how='inner', on='Category')
df_events = pd.merge(df_events, zipcode_mapping, how='inner', left_on='Zip code', right_on='zip')
df_events = pd.merge(df_events, state_mapping, how='inner', left_on='state', right_on='Region Code')
df_events = pd.merge(df_events, state_wages, how='inner', left_on='Region Code', right_on='State Code')

# ## Exclude Other Events since they are small and not clear which events they are
df_events = df_events[df_events['Grouped Category'] != 'other']
df_events.isnull().sum()
df_events.shape

# ## This is a pie chart for distribution of events which added donation feature and didn't add donation feature. There is a big class imbalance which shows 1.73% of events added donation features
import plotly.graph_objs as go

don = df_events['Donation'].value_counts()

layout = go.Layout(
    title="Donation Added/Not-Added (2017 Jan -2020 Apr)") 
fig = go.Figure(data=[go.Pie(labels=['W/O Donations','W/ Donations'],
                             values=don.values)], layout = layout)
fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,
                  marker=dict(colors=['#E0E0E0','#BD2D2D'], line=dict(color='#000000', width=2)))
fig.show()


# ## This is a scatter plot for how Goals look like for each event's (that added donation) raised money. There is not much correlation between Goals and Total Raised for Events
# x and y given as DataFrame columns
# Goal of the event doesn't show clear linear relationship with Donation Amount
import plotly.express as px
events2 = events.loc[events['Goal'] <= 30000.0,:]
events2 = events2.loc[events2['Goal'] > 0.0,:]
fig = px.scatter(events2, x="Goal", y="Total Raised")
fig.show()

# ## Color Scheme to use throughout the Graphs
colors = [
      '#BD2D2D','#AF4C39', #red
      '#F16301','#FF6F91','#FF9671','#FFC75F','#F9F871', #Yellow, Orange
      '#B8EF83','#4BCEAE','#3FA2B4','#145A32', #Green
      '#ABBFF6','#3C8AF2','#2471A3','#154360', #Blue
      '#EBDEF0','#C999CA','#B574FB','#8B008B','#581845', #Pueple
      '#99C0DB',
      "#c7785e","#eebe76","#674942","#422929","#d6ab8e"]
colors2 = np.repeat('#3C8AF2',100).tolist()
colors = colors + colors2

# ## Frequency Distribution for each Event Category and Grouped Event Category
category = df_events['Category'].value_counts()

layout = go.Layout(
    title="Frequency Distribution of Categories (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="Categories"
    ),
    yaxis=dict(
        title="Frequency"
    ) ) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()

category = df_events['Grouped Category'].value_counts()

layout = go.Layout(
    title="Frequency Distribution of Grouped Categories (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="Grouped Categories"
    ),
    yaxis=dict(
        title="Frequency"
    ) ) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()

# ## For Grouped Categories, how much portion added donation features? Organizations, Weddings, Design_own, and Birthday Parties were more likely to add donation features.
category1 = df_events[df_events['Donation'] == 1]
category_percent = round(category1['Grouped Category'].value_counts()/df_events['Grouped Category'].value_counts()*100, 2)
category_percent = category_percent[category.index]

layout = go.Layout(
    title="Percentage(%) to Add Donations for Grouped Categories (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="Grouped Categories"
    ),
    yaxis=dict(
        title="Percentage(%) to Add Donations"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 1.73, # use absolute value or variable here
            'x1': 1,
            'y1': 1.73, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category_percent.index,
    y=category_percent.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## Frequency Distribution and Percentage that added donation feature for States
category = df_events['Region Code'].value_counts()

layout = go.Layout(
    title="Frequency Distribution of States (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="States"
    ),
    yaxis=dict(
        title="Frequency"
    ) ) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()

category1 = df_events[df_events['Donation'] == 1]
category_percent = round(category1['Region Code'].value_counts()/df_events['Region Code'].value_counts()*100, 2)
category_percent = category_percent[category.index]
layout = go.Layout(
    title="Percentage(%) to Add Donations for States (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="States"
    ),
    yaxis=dict(
        title="Percentage(%) to Add Donations"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 1.73, # use absolute value or variable here
            'x1': 1,
            'y1': 1.73, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category_percent.index,
    y=category_percent.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## Annual Mean Wages for each state in 2018. High Wage States are likely to have more percentage of events to add donation features. CA, NY, IL, WA, OR, DC, etc
stateWage = state_wages.set_index('State Code')
stateWage = stateWage['Annual Mean Wage']
stateWage = stateWage[category.index]

layout = go.Layout(
    title="Annual Mean Wage for States (2018)",
    xaxis=dict(
        title="States"
    ),
    yaxis=dict(
        title="Annual Mean Wage($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 1.73, # use absolute value or variable here
            'x1': 1,
            'y1': 1.73, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=stateWage.index,
    y=stateWage.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## Frequency Distribution and Percentage that added donation feature for Region. South had most events followed by West, Midwest, and Northeast. However, West and Northeast had more portion of events that added donations.
category = df_events['Grouped Region'].value_counts()

layout = go.Layout(
    title="Frequency Distribution of Regions (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="Regions"
    ),
    yaxis=dict(
        title="Frequency"
    ) ) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()

category1 = df_events[df_events['Donation'] == 1]
category_percent = round(category1['Grouped Region'].value_counts()/df_events['Grouped Region'].value_counts()*100, 2)
category_percent = category_percent[category.index]
layout = go.Layout(
    title="Percentage(%) to Add Donations for Regions (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="Regions"
    ),
    yaxis=dict(
        title="Percentage(%) to Add Donations"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 1.73, # use absolute value or variable here
            'x1': 1,
            'y1': 1.73, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category_percent.index,
    y=category_percent.values,
    marker_color=colors
)], layout = layout)
fig.show()

np.mean(df_events['Annual Mean Wage'])


# ## West and Northeast have higher wages and these regions add more donation features. There is strong correlation with Average Wages and Proportion of Events to add donation features.
category2 = df_events.groupby('Grouped Region')['Annual Mean Wage'].apply(np.mean)
category2 = category2[category.index]

layout = go.Layout(
    title="Average Wage For Regions (2018)",
    xaxis=dict(
        title="Grouped Region"
    ),
    yaxis=dict(
        title="Average Wage($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 53000, # use absolute value or variable here
            'x1': 1,
            'y1': 53000, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category2.index,
    y=category2.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## Frequency Distribution and Percentage that added donation feature for each quarter Q1 to Q4. There were a lot more events in Q4.
df_events['Quarter'] = "Q" + df_events['Quarter'].astype(str)
category = df_events['Quarter'].value_counts()

layout = go.Layout(
    title="Frequency Distribution of Quarters (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="Quarters"
    ),
    yaxis=dict(
        title="Frequency"
    ) ) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## Q1 and Q4 had more events that added donation features.
category1 = df_events[df_events['Donation'] == 1]
category_percent = round(category1['Quarter'].value_counts()/df_events['Quarter'].value_counts()*100, 2)
category_percent = category_percent[category.index]
layout = go.Layout(
    title="Percentage(%) to Add Donations for Quarters (2017 Jan -2020 Apr)",
    xaxis=dict(
        title="Quarters"
    ),
    yaxis=dict(
        title="Percentage(%) to Add Donations"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 1.73, # use absolute value or variable here
            'x1': 1,
            'y1': 1.73, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category_percent.index,
    y=category_percent.values,
    marker_color=colors
)], layout = layout)
fig.show()

# ## Observe how frequency of Events differs for each quarter. Birthday Parties were noticeably the highest number, but other events had different orders for each quarter.
GroupCat = pd.unique(df_events['Quarter'])
for i in GroupCat:
    df_GroupCat = df_events[df_events['Quarter'] == i]
    sampleSize = len(df_GroupCat)
    if (sampleSize > 5000):
        category1 = df_GroupCat[df_GroupCat['Donation'] == 1]
        category_percent = category1['Grouped Category'].value_counts()

        layout = go.Layout(
            title="(2017 Jul -2020 Mar) Quarter: " + i + ", Frequency",
            xaxis=dict(
                title="Grouped Category"
            ),
            yaxis=dict(
                title="Frequency"
            ),
            shapes=[{
                'type': 'line',
                'xref': 'paper',
                'x0': 0,
                'y0': 1.73, # use absolute value or variable here
                'x1': 1,
                'y1': 1.73, # ditto
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 1,
                    'dash': 'dash',
                    },
                }
                ]) 
        fig = go.Figure(data=[go.Bar(
            x=category_percent.index,
            y=category_percent.values,
            marker_color=colors
        )], layout = layout)
        fig.show()

df_events.isnull().sum()


# ## For modeling purpose, need to encode dummy variables for all the categorical variables that are going into the model such as Quarter, Grouped Category, Grouped Region
# Create Dummy Variables
events_fn = pd.get_dummies(df_events, columns=['Quarter','Grouped Category','Grouped Region'])
todrop = ['Event ID','Category','Event Date','Zip code','zip','state','Region Code','Median Monthly Rent','Value of a Dollar','Annual Adjusted Mean Wage','State','State Code']
events_fn = events_fn.drop(columns = todrop)
events_fn.columns.values


# ## To prevent problems from class imbalance, SMOTE algorithm is needed to sample data for events that add donation features so that we have balanced dataset
# SMOTE algorithm(Synthetic Minority Oversampling Technique)
# Works by creating synthetic samples from the minor class (donation feature) instead of creating copies.
# Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations.
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X = events_fn.loc[:, events_fn.columns != 'Donation']
y = events_fn.loc[:, events_fn.columns == 'Donation']
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
os_data_y= pd.DataFrame(data=os_data_y,columns=['Donation'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of No Donation in oversampled data",len(os_data_y[os_data_y['Donation']==0]))
print("Number of Donation",len(os_data_y[os_data_y['Donation']==1]))
print("Proportion of No Donation data in oversampled data is ",len(os_data_y[os_data_y['Donation']==0])/len(os_data_X))
print("Proportion of Donation data in oversampled data is ",len(os_data_y[os_data_y['Donation']==1])/len(os_data_X))
# Perfectly balanced data now

# Recursive Feature Elimination
# to repeatedly construct a model and choose either the best or worst performing feature, 
# setting the feature aside and then repeating the process with the rest of the features
data_final_vars=events_fn.columns.values.tolist()
y=['Donation']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

# ## Go through Statistical Analysis to exclude some features that are statistically insignificant in the model
# All features are selected
X=os_data_X
y=os_data_y['Donation']

# Implement the model
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

# ## The model result suggests the features with less numbers are causing an event less likely to add donation feature. All the following obervations are statistically significant and proven. For example, Events in Q1 and Q4 are more likely to add donation features than Q3 and Q2. Events for Organizations and Weddings and Design_Own and Birthday parties are more likely to add donation features than other events for like Babies/Kids and Spanish.  Events from West and Northeast are more likely to add donation features than other two regions. This model result is aligned with the exploratory data analysis that was done before modeling phase.
X=os_data_X.drop(columns=['Annual Mean Wage'])
y=os_data_y['Donation']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

# ## Fit a logistic regression model using 75% of Training set and validate the model with 25% of Test set.
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# ## Accuracy of prediction is 58%
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


# ## Confusion matrix to see the result of prediction for the entire test dataset

# Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes = ['No Donation', 'Donation'],
                      title = 'Donation Feature Confusion Matrix')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# ## The ROC curve is not too curvy which means the model is not a good model in predicting
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Time-series Forecasting on Donation Amount over time
donations.isnull().sum()
org.isnull().sum()
events.tail()
org.tail()
donations.tail()

# Time Series for Donations (ARIMA)
df_Time = donations
df_Time['Date'] = pd.to_datetime(df_Time['Date'], errors = 'coerce', utc=True)
df_Time = df_Time.resample('W-Sun', on='Date').sum().reset_index().sort_values(by='Date')[['Date','Donation Amount']]
df_Time.head()


# ## Graph for Donation over Time from 2015 to 2020. There is a seasonality trend over time
layout = go.Layout(
    title="Donation over time(2015 Sep -2020 Mar)",
    xaxis=dict(
        title="Week"
    ),
    yaxis=dict(
        title="Donation Amount($)"
    ))
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=df_Time.Date, y=df_Time['Donation Amount'],
                         line=dict(color=colors[0], width=4)))
fig.show()

df_Time = df_Time[df_Time['Date'] <= '2020-03-01']
df_Time = df_Time.reset_index()

from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df_Time['Donation Amount'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# p-value for ADF statistics is bigger than 0.05 so you need differencing

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df_Time['Donation Amount']); axes[0, 0].set_title('Original Series')
plot_acf(df_Time['Donation Amount'], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df_Time['Donation Amount'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df_Time['Donation Amount'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df_Time['Donation Amount'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df_Time['Donation Amount'].diff().diff().dropna(), ax=axes[2, 1])

plt.show()
# d = 1

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df_Time['Donation Amount'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,2))
plot_pacf(df_Time['Donation Amount'].diff().dropna(), ax=axes[1])

plt.show()
# p is 0

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df_Time['Donation Amount'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df_Time['Donation Amount'].diff().dropna(), ax=axes[1])

plt.show()
# q is 0

# Build ARIMA with p,d,q
from statsmodels.tsa.arima_model import ARIMA

# 0,1,0 ARIMA Model
model = ARIMA(df_Time['Donation Amount'], order=(0,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()

print(len(df_Time))
print(len(df_Time)*0.75)

from statsmodels.tsa.stattools import acf

# Create Training and Test
train = df_Time['Donation Amount'][:175]
test = df_Time['Donation Amount'][175:]

# Build Model
# model = ARIMA(train, order=(3,2,1))  
model = ARIMA(train, order=(1, 0, 1))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(234-175, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Try Auto-Arima to figure out p,d,q
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

model = pm.auto_arima(df_Time['Donation Amount'], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(7,5))
plt.show()

# ## Forecast model without Seasonality
# Forecast
n_periods = 59
fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df_Time['Donation Amount']), len(df_Time['Donation Amount'])+n_periods)

fc_series = pd.Series(fitted, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(df_Time['Donation Amount'], label='Original Series')
axes[0].plot(df_Time['Donation Amount'].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

# Seasinal Dei
axes[1].plot(df_Time['Donation Amount'], label='Original Series')
axes[1].plot(df_Time['Donation Amount'].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Donation Amount', fontsize=16)
plt.show()

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(train, start_p=1, start_q=1,
                         test='adf',
                         max_p=4, max_q=4, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()


# ## Forecast model with Seasonality
# Forecast
n_periods = 59
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df_Time['Donation Amount']), len(df_Time['Donation Amount'])+n_periods)

fc_series = pd.Series(fitted, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# ## Final Forecast Model with ARIMA technique. There is not much insight generated from this modeling unfortunately besides that the general trend of total donation amount is going up with more events over time.
# Forecast
n_periods = 39
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df_Time['Donation Amount']), len(df_Time['Donation Amount'])+n_periods)

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df_Time['Donation Amount'])
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMA - Final Forecast of Donation Amount")
plt.show()


# # Prediction of Donation Amount for Events that added donation features

# ## Merge Donation and Events Tables
# Data for Donations/Events/Organizations (Linear Regression)
events = pd.read_csv('evite-reports/events.csv')
events = events[events['Beneficiary Type'] == 'Organization']
df = pd.merge(donations, events, how='left', on='Event ID')

df.shape
df.isnull().sum()
df.head()

# ## Clean column names since there are two Nonprofit ID columns.
df['Nonprofit ID'] = np.where(df['Nonprofit ID_x'].isna() , df['Nonprofit ID_y'], df['Nonprofit ID_x'])
df = df.drop(columns = ['Nonprofit ID_x','Nonprofit ID_y'])

for col in df.columns: 
    print(col) 


# ## Join Two joined tables with Organization Table
df = pd.merge(df, org, how='left', left_on ='Nonprofit ID', right_on = 'ID')

df.isnull().sum()

print('org_mapping:')
for col in org_mapping.columns: 
    print(col)
print("")
print('state_mapping:')
for col in state_mapping.columns: 
    print(col)


# # Mapping for Grouped Category, Region, and this time Grouped Organization (donation goes to)
df_final = df
df_final = pd.merge(df_final, event_mapping, how='inner', on='Category')
df_final = pd.merge(df_final, state_mapping, how='inner', on='Region Code')
df_final = pd.merge(df_final, org_mapping, how='inner', on='Cause')


# ## Exclude "Other" events and filter for US events. 
# Other Grouped Category is skewed
df_final = df_final[df_final['Grouped Category'] != 'other']
# Filter for only US data
df_final = df_final[df_final['Country Code'] == "US"]
# Create Quarter
df_final['Date'] = pd.to_datetime(df_final['Date'], errors = 'coerce', utc=True)
df_final['Month'] = df_final['Date'].dt.month
df_final['Quarter'] = df_final['Date'].dt.quarter


# ## This cleaning is for creating Time-series Charts for each Grouped Category
df_Time = df_final.groupby('Grouped Category').resample('W-Sun', on='Date').sum().reset_index().sort_values(by='Date')
df_Time.head()
df_Time = df_Time.pivot(index='Date',columns='Grouped Category',values='Donation Amount')
df_Time.head()
df_Time.fillna(0, inplace=True)

# ## For each Category, Peaks are in different periods. Birthday parties have peaks in Spring and Dips in Summer. Fall Winter peaks in fall and winter, while spring summer peaks in spring and summer obviously. Organizations and Get Togethers generally have been rising and increased in 2019 the most. Weddings peak in the summer for obvious weather reasons.
layout = go.Layout(
    title="Donation For Categories over time (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Week"
    ),
    yaxis=dict(
        title="Donation Amount($)"
    ))
fig = go.Figure(layout=layout)
for i in range(len(df_Time.columns)):
    fig.add_trace(go.Scatter(x=df_Time.index, y=df_Time.iloc[:,i], name=df_Time.columns[i],
                         line=dict(color=colors[i], width=4)))
fig.show()


# ## Time series trends for each Grouped Donation Cause. There is a huge increase for Public Society and Health causes in 2019. For these causes, peaks are located in May and December mostly, and then dip in the summer and after December.
df_Time = df_final.groupby('Grouped Cause').resample('W-Sun', on='Date').sum().reset_index().sort_values(by='Date')
df_Time = df_Time.pivot(index='Date',columns='Grouped Cause',values='Donation Amount')
df_Time.fillna(0, inplace=True)
layout = go.Layout(
    title="Donation For Causes over time (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Week"
    ),
    yaxis=dict(
        title="Donation Amount($)"
    ))
fig = go.Figure(layout=layout)
for i in range(len(df_Time.columns)):
    fig.add_trace(go.Scatter(x=df_Time.index, y=df_Time.iloc[:,i], name=df_Time.columns[i],
                         line=dict(color=colors[i], width=4)))
fig.show()

# ## Drop irrelevant columns that won't be used in the regression modeling
df_final = df_final.drop(columns = ['Event Date','Internal ID','Zip code','Host email','Total Raised','Total donors','State','Zip','Country','EIN','Created At'])
df_final = df_final.drop(columns = ['User ID','Email','Event ID','Nonprofit ID','Postal Code','Tip Amount','Country Code','Beneficiary Type','ID'])

# ## Count the number of NA values in the dataset. Only 39 rows of data have NA in Goal, so replace those NA with 0
df_final.isnull().sum()
df_final['Goal'].fillna(0, inplace=True)
df_final.dtypes
df_final.shape

df_final['Donation Amount'].skew()
df_final.shape
df_final['Donation Amount'].skew()
df_final.dtypes


# ## See Distribution of Donation Amount while excluding some outliers. Range of the graph will be Donation Amount less than 1000 dollars, and Goal is less than 10000 dollars. There is an obvious skewed distribution since an individual donator wouldn't donate more than 1000 dollars.
df_final2 = df_final.loc[df_final['Donation Amount'] <= 1000.0,:]
df_final2 = df_final2.loc[df_final2['Goal'] <= 10000.0,:]

plt.hist(df_final2['Donation Amount'], bins = 500)
plt.xlabel('Donation Amount')
plt.ylabel('Frequency')
plt.title('Histogram for Donation Amount (2017 Jul to 2020 Mar)')
plt.show()

fig = go.Figure(data=[go.Histogram(x=df_final2['Donation Amount'])])
fig.show()

# ## Goal and Donation Amount are not much correlation with each other based on the scatter plot.
# x and y given as DataFrame columns
# Goal of the event doesn't show clear linear relationship with Donation Amount
import plotly.express as px
fig = px.scatter(df_final2, x="Goal", y="Donation Amount")
fig.show()

# ## This Overall Average Donation Amount will be used as a guide line in each graph.
df_final['Donation Amount'].mean()

# ## Graph For Average Donation for Categories. It doesn't have information on the number of events for each category, so high numbers can be biased such as ramadan.
category = df_final.groupby('Category')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Categories (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Categories"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## Average Donations for Organizations and Weddings and Get Togethers were higher than other Groups. Organizations and Weddings are aligned with the first subject of study, but Get Togethers' high donation amount is new insight. Additionally, birthday parties had lower average donation amounts than other groups. Birthday parties were more likely to add donation features but due to the nature of the events, event participants did not donate as much.
category = df_final.groupby('Grouped Category')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Grouped Categories (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Grouped Categories"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## Except for certain states MT and WY, general trend for average donations for states was similar to the first study. The states with higher state wages added more donations in average.
category = df_final.groupby('Region Code')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For States (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="States"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()

stateWage = state_wages.set_index('State Code')
stateWage = stateWage['Annual Mean Wage']
stateWage = stateWage[category.index]

layout = go.Layout(
    title="Annual Mean Wage for States (2018)",
    xaxis=dict(
        title="States"
    ),
    yaxis=dict(
        title="Annual Mean Wage($)"
    )) 
fig = go.Figure(data=[go.Bar(
    x=stateWage.index,
    y=stateWage.values,
    marker_color=colors
)], layout = layout)
fig.show()

# ## Just like the first study, West was the State with the highest average donation amount followed by Northeast. This is aligned with the previous results that regions with higher average incomes were more likely to add more donation amounts to events.
category = df_final.groupby('Grouped Region')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Grouped Regions (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Grouped Regions"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## There was an interesting insight from the graph that Science&Technology and College Organizations received the most donation amounts in average. Another interesting insight was people did not donate as much to Animal organizations. This might be due to the number of events that donated to Animal Organizations 
category = df_final.groupby('Cause')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Different Causes (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Cause"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()
#  ## Grouped Causes: Arts and Science had the biggest donation amount consistent with the previous result that events donating to Science organizations were able to generate a lot more donations than other types of causes. On the other hand, events that donated to Animals Organizations were, consistent with the previous result as well, donating less than for other causes.
category = df_final.groupby('Grouped Cause')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Grouped Causes (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Grouped Cause"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## The type of events that contributed to high donations for Science Organizations was meeting and book-club. It means these meetings were science-related and able to generate a lot more donations for Science causes.
arts_science = df_final[df_final['Grouped Cause'] == 'Arts & Science']
arts_science = arts_science.groupby('Category')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Category for Arts & Science (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Category"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=arts_science.index,
    y=arts_science.values,
    marker_color=colors
)], layout = layout)
fig.show()

animals = df_final[df_final['Grouped Cause'] == 'Animals & Environment']
animals = animals.groupby('Category')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Category for Animals & Environment (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Category"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=animals.index,
    y=animals.values,
    marker_color=colors
)], layout = layout)
fig.show()


# ## Average donation amount for each month. This was surprisingly inconsistent with the first study. The first study suggested Q1 and Q4 had the higher probability of adding donation features for events. However, from this graph, it indicates that events are donation more money during the second half of the year. Q1 had the least amount of donation in average.

df_final['Month'] = df_final['Month'].astype(str)
category = df_final.groupby('Month')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Month (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Month"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()

df_final['Quarter'] = "Q" + df_final['Quarter'].astype(str)
category = df_final.groupby('Quarter')['Donation Amount'].apply(np.mean)

layout = go.Layout(
    title="Average Donation For Quarters (2017 Jul -2020 Mar)",
    xaxis=dict(
        title="Quarter"
    ),
    yaxis=dict(
        title="Average Donation($)"
    ),
    shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
fig = go.Figure(data=[go.Bar(
    x=category.index,
    y=category.values,
    marker_color=colors
)], layout = layout)
fig.show()

df_final['Grouped Category'].value_counts()
pd.unique(df_final['Grouped Category'])


# ## This set of graphs is to demonstrate what kind of causes events donated to generated more donations for each type of events.
GroupCat = pd.unique(df_final['Grouped Category'])
for i in GroupCat:
    df_GroupCat = df_final[df_final['Grouped Category'] == i]
    sampleSize = len(df_GroupCat)
    df_GroupCat = df_GroupCat.groupby('Cause')['Donation Amount'].apply(np.mean)

    layout = go.Layout(
        title="(2017 Jul -2020 Mar) Event: " + i + ", Number of Donations: " + str(sampleSize),
        xaxis=dict(
            title="Causes"
        ),
        yaxis=dict(
            title="Average Donation($)"
        ),
        shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
    fig = go.Figure(data=[go.Bar(
        x=df_GroupCat.index,
        y=df_GroupCat.values,
        marker_color=colors
    )], layout = layout)
    fig.show()


# ## ## This set of graphs is to demonstrate what kind of causes events donated to generated more donations for each state. Only includes states that had more than 5000 events.
GroupCat = pd.unique(df_final['Region Code'])
for i in GroupCat:
    df_GroupCat = df_final[df_final['Region Code'] == i]
    sampleSize = len(df_GroupCat)
    if (sampleSize > 5000):
        df_GroupCat = df_GroupCat.groupby('Cause')['Donation Amount'].apply(np.mean)

        layout = go.Layout(
            title="(2017 Jul -2020 Mar) Region: " + i + ", Number of Donations: " + str(sampleSize),
            xaxis=dict(
                title="Cause"
            ),
            yaxis=dict(
                title="Average Donation($)"
            ),
            shapes=[{
                'type': 'line',
                'xref': 'paper',
                'x0': 0,
                'y0': 48.5, # use absolute value or variable here
                'x1': 1,
                'y1': 48.5, # ditto
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 1,
                    'dash': 'dash',
                    },
                }
                ]) 
        fig = go.Figure(data=[go.Bar(
            x=df_GroupCat.index,
            y=df_GroupCat.values,
            marker_color=colors
        )], layout = layout)
        fig.show()


# ## For each region out of 4 different regions, there are different donation causes that generated most donations from events. For South: Literacy and College related donations. For West, Science-related donations by far. For Northeast, College related and Autism and HIV related donations. For Midwest, Humanitarian Assistance and HIV and Autism related donations. Interestingly, most regions had college related causes with high donations, and HIV and Autism, but South had relatively low donations to HIV/Autism-related causes.
GroupCat = pd.unique(df_final['Grouped Region'])
for i in GroupCat:
    df_GroupCat = df_final[df_final['Grouped Region'] == i]
    sampleSize = len(df_GroupCat)
    df_GroupCat = df_GroupCat.groupby('Cause')['Donation Amount'].apply(np.mean)

    layout = go.Layout(
        title="(2017 Jul -2020 Mar) Region: " + i + ", Number of Donations: " + str(sampleSize),
        xaxis=dict(
            title="Cause"
        ),
        yaxis=dict(
            title="Average Donation($)"
        ),
        shapes=[{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'y0': 48.5, # use absolute value or variable here
            'x1': 1,
            'y1': 48.5, # ditto
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 1,
                'dash': 'dash',
                },
            }
            ]) 
    fig = go.Figure(data=[go.Bar(
        x=df_GroupCat.index,
        y=df_GroupCat.values,
        marker_color=colors
    )], layout = layout)
    fig.show()

# ## Data is reordered in descending order by donation amount
df_final.sort_values(by='Donation Amount', ascending=False)
df_final.columns

# ## Like the first study, categorical variables have to be encoded to dumym variables.
# Create Dummy Variables
df_final_input = pd.get_dummies(df_final, columns=['Quarter','Grouped Category','Grouped Region','Grouped Cause'])

#df_final_input = df_final_input[df_final_input['Date'] >= '2019-01-01']
todrop = ['Category','Cause','Name','Region Code','Month','Date']
df_final_input = df_final_input.drop(columns = todrop)
df_final_input.columns.values

X = df_final_input.loc[:, df_final_input.columns != 'Donation Amount']
y = df_final_input.loc[:, df_final_input.columns == 'Donation Amount']

# Implement the model
import statsmodels.api as sm
reg_model=sm.OLS(y,X)
result=reg_model.fit()
print(result.summary())

X=X.drop(columns=['Grouped Category_spanish'])
reg_model=sm.OLS(y,X)
result=reg_model.fit()
print(result.summary())

# ## After removing some features that are statistically insignificant in the model, Coefficients are telling relevant stories about how each feature is contributing to the projected donation amount. One noticeable insight is birthday parties are definitely negatively contributing to the donation amount for each event. Organizations events, Region West, and Science Causes all lead to more donations from an event.

X=X.drop(columns=['Grouped Category_babies_kids','Grouped Category_fall_winter','Grouped Category_spring_summer'])
reg_model=sm.OLS(y,X)
result=reg_model.fit()
print(result.summary())

from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
regr = LinearRegression()  
regr.fit(X_train, y_train) #training the algorithm

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

y_pred = regr.predict(X_test)


# ## Errors are high which means the model is not good at predicting donation amounts.
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def graph(formula, x_range, label=None):
    """
    Helper function for plotting cook's distance lines
    """
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')


def diagnostic_plots(X, y, model_fit=None):
  """
  Function to reproduce the 4 base plots of an OLS model in R.

  ---
  Inputs:

  X: A numpy array or pandas dataframe of the features to use in building the linear regression model

  y: A numpy array or pandas series/dataframe of the target variable of the linear regression model

  model_fit [optional]: a statsmodel.api.OLS model after regressing y on X. If not provided, will be
                        generated from X, y
  """

  if not model_fit:
      model_fit = sm.OLS(y, sm.add_constant(X)).fit()

  # create dataframe from X, y for easier plot handling
  dataframe = pd.concat([X, y], axis=1)

  # model values
  model_fitted_y = model_fit.fittedvalues
  # model residuals
  model_residuals = model_fit.resid
  # normalized residuals
  model_norm_residuals = model_fit.get_influence().resid_studentized_internal
  # absolute squared normalized residuals
  model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
  # absolute residuals
  model_abs_resid = np.abs(model_residuals)
  # leverage, from statsmodels internals
  model_leverage = model_fit.get_influence().hat_matrix_diag
  # cook's distance, from statsmodels internals
  model_cooks = model_fit.get_influence().cooks_distance[0]

  plot_lm_1 = plt.figure()
  plot_lm_1.axes[0] = sns.residplot(model_fitted_y, dataframe.columns[-1], data=dataframe,
                            lowess=True,
                            scatter_kws={'alpha': 0.5},
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

  plot_lm_1.axes[0].set_title('Residuals vs Fitted')
  plot_lm_1.axes[0].set_xlabel('Fitted values')
  plot_lm_1.axes[0].set_ylabel('Residuals');

  # annotations
  abs_resid = model_abs_resid.sort_values(ascending=False)
  abs_resid_top_3 = abs_resid[:3]
  for i in abs_resid_top_3.index:
      plot_lm_1.axes[0].annotate(i,
                                 xy=(model_fitted_y[i],
                                     model_residuals[i]));

  QQ = ProbPlot(model_norm_residuals)
  plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
  plot_lm_2.axes[0].set_title('Normal Q-Q')
  plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
  plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
  # annotations
  abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
  abs_norm_resid_top_3 = abs_norm_resid[:3]
  for r, i in enumerate(abs_norm_resid_top_3):
      plot_lm_2.axes[0].annotate(i,
                                 xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                     model_norm_residuals[i]));

  plot_lm_3 = plt.figure()
  plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);
  sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
  plot_lm_3.axes[0].set_title('Scale-Location')
  plot_lm_3.axes[0].set_xlabel('Fitted values')
  plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

  # annotations
  abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
  abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
  for i in abs_norm_resid_top_3:
      plot_lm_3.axes[0].annotate(i,
                                 xy=(model_fitted_y[i],
                                     model_norm_residuals_abs_sqrt[i]));


  plot_lm_4 = plt.figure();
  plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);
  sns.regplot(model_leverage, model_norm_residuals,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
  plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
  plot_lm_4.axes[0].set_ylim(-3, 5)
  plot_lm_4.axes[0].set_title('Residuals vs Leverage')
  plot_lm_4.axes[0].set_xlabel('Leverage')
  plot_lm_4.axes[0].set_ylabel('Standardized Residuals');

  # annotations
  leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
  for i in leverage_top_3:
      plot_lm_4.axes[0].annotate(i,
                                 xy=(model_leverage[i],
                                     model_norm_residuals[i]));

  p = len(model_fit.params) # number of model parameters
  graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
        np.linspace(0.001, max(model_leverage), 50),
        'Cook\'s distance') # 0.5 line
  graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
        np.linspace(0.001, max(model_leverage), 50)) # 1 line
  plot_lm_4.legend(loc='upper right');


# ## Some linear regression diagnostic graphs checking assumptions
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
diagnostic_plots(X, y)

