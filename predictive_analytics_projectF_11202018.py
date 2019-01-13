#https://www.dataoptimal.com/data-cleaning-with-python-2018/
import pandas as pd
gift= pd.read_csv(r'gifts.csv',sep=';')
donors= pd.read_csv(r'donors.csv',sep=';')
train= pd.read_csv(r'campaign20130411.csv',sep=';')
test= pd.read_csv(r'campaign20140115.csv',sep=';')
import numpy as np

#cleaning donor dataset
#donors.isnull().values.any()
# count the number of NaN values in each column
#sum(donors['region']=='Missing')/len(donors)
#78% of the data is missing. we will drop the column. Also we can refer to the region

donors.head() 
donors['zipcode'] = pd.to_numeric(donors['zipcode'], errors = 'coerce')    
#categorizing regions by zipcode
donors["Capital_Region"] = ((donors["zipcode"]>=1000) & (donors["zipcode"]<1300)).astype(int)
donors["Walloon"] = ((donors["zipcode"]>=1300) & (donors["zipcode"]<1499)).astype(int)
donors["Flemish"] = ((donors["zipcode"]>=1500) & (donors["zipcode"]<1999)).astype(int)
donors["Antwerp"] = ((donors["zipcode"]>=2000) & (donors["zipcode"]<2999)).astype(int)
donors["Flemish2"] = ((donors["zipcode"]>=3000) & (donors["zipcode"]<3499)).astype(int)
donors["Limburg"] = ((donors["zipcode"]>=3500) & (donors["zipcode"]<3999)).astype(int)
donors["Liege"] = ((donors["zipcode"]>=4000) & (donors["zipcode"]<4999)).astype(int)
donors["Namur"] = ((donors["zipcode"]>=5000) & (donors["zipcode"]<5999)).astype(int)
donors["Hainaut"] = ((donors["zipcode"]>=6000) & (donors["zipcode"]<6599)).astype(int)
donors["Luxembourg"] = ((donors["zipcode"]>=6600) & (donors["zipcode"]<6999)).astype(int)
donors["Hainaut2"] = ((donors["zipcode"]>=7000) & (donors["zipcode"]<7999)).astype(int)
donors["West_F"] = ((donors["zipcode"]>=8000) & (donors["zipcode"]<8999)).astype(int)
donors["East_F"] = ((donors["zipcode"]>=9000) & (donors["zipcode"]<9999)).astype(int)

del donors['region']

#making gender and language columns binary
dummy=pd.get_dummies(data=donors, columns=['gender','language'])

#5 year► time frame for the gift train/test dataset
gift['date'] = pd.to_datetime(gift['date'])
Gifts_test = gift[(gift['date'] >= '01/01/2009') & (gift['date'] < '01/01/2014')]
Gifts_train = gift[(gift['date'] >= '01/01/2008') & (gift['date'] < '01/01/2013')]

#gifts datatables groupby donors from the last five years
lastfive_gifts_train = Gifts_train.groupby(['donorID'], as_index=False).sum() 
lastfive_gifts_test = Gifts_test.groupby(['donorID'], as_index=False).sum() 

#final base tables 
basetable_train = pd.merge(dummy,lastfive_gifts_train,how="outer") 
train["donated"] = ((train["amount"]>0)).astype(int)
del train["amount"]
basetable_train_f = pd.merge(train,basetable_train) 
del basetable_train_f["campID"]

basetable_test = pd.merge(dummy,lastfive_gifts_test,how="outer") 
test["donated"] = ((test["amount"]>0)).astype(int)
del test["amount"]
basetable_test_f = pd.merge(test,basetable_test) 
del basetable_test_f["campID"]

# rename amount column to amount5yrs; it is amount donated in last 5 years
basetable_train_f = basetable_train_f.rename(columns={'amount': 'amount5yrs'})
basetable_test_f = basetable_test_f.rename(columns={'amount': 'amount5yrs'})

########################################################################################################ch1 Building Logistic Regression Models
#Exploring the base table
population_size  = len(basetable_train_f) #34917
targets_count = sum(basetable_train_f["donated"]) #122
print(targets_count/population_size) #0.03494000057278689; incidence #3.4 percent donated
basetable_train_f["amount5yrs"].fillna(0, inplace=True)
basetable_test_f["amount5yrs"].fillna(0, inplace=True)

#Building a logistic regression model
from sklearn import linear_model
X = basetable_train_f[["Capital_Region","gender_F","language_F","amount5yrs"]]
y = basetable_train_f[["donated"]] #donated is the target
logreg = linear_model.LogisticRegression()
logreg.fit(X, y)

#Showing the coefficients and intercept
predictors = ["Capital_Region","gender_F","language_F","amount5yrs"]
X = basetable_train_f[predictors]
y = basetable_train_f[["donated"]]
logreg = linear_model.LogisticRegression()
logreg.fit(X, y)

# Assign the coefficients to a list coef
coef = logreg.coef_
for p,c in zip(predictors,list(coef[0])):
    print(p + '\t' + str(c))

# Assign the intercept to the variable intercept
intercept = logreg.intercept_
print(intercept) 

#Making predictions
# Create a dataframe new_data from current_data that has only the relevant predictors 
new_data = basetable_train_f[["Capital_Region","gender_F","language_F","amount5yrs"]]
predictions = logreg.predict_proba(new_data)
print(predictions[0:5])
#[[0.96736216 0.03263784]
# [0.97134251 0.02865749]
# [0.97402751 0.02597249]
# [0.96736216 0.03263784]
# [0.96870314 0.03129686]]

#Donor that is most likely to donate
#Sort the predictions
predictions[0].astype(np.int64)
predictions[1].astype(np.int64)
#predictions_sorted = predictions.sort(["probability"])

##############################################################################################################ch2 Forward stepwise variable selection for logistic regression

#Calculating model attributes
import numpy as np
from sklearn.metrics import roc_auc_score

#roc_auc_score(true_target, prob_target)
predictions = logreg.predict_proba(X)
predictions_target = predictions[:,1]

# Calculate the AUC value
auc = roc_auc_score(y, predictions_target)
print(round(auc,2))

#Using different variables; stepwise
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

def auc(X, y, basetable_train_f):

    X = basetable_train_f[predictors]
    y = basetable_train_f[["donated"]]
  
    logreg = linear_model.LogisticRegression()
    logreg.fit(X, y)
    predictions = logreg.predict_proba(X)[:,1]
    auc = roc_auc_score(y, predictions)
    return(auc)
    
print(auc)

#AUC CURVES !!!
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, predictions_target)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


#creating some new variables from gifst dataset
min_amount5yrs = basetable_train_f['amount5yrs'].min()
mean_amount5yrs = basetable_train_f['amount5yrs'].mean()
max_amount5yrs = basetable_train_f['amount5yrs'].max()

#finding next best variablesto include in the model
current_variables = ["Capital_Region","gender_F","language_F","amount5yrs"]
candidate_variables = ["min_amount5yrs","max_amount5yrs","mean_amount5yrs"]
target = y

def next_best(current_variables,candidate_variables, target, basetable_train_f):
    
    best_auc = -1
    best_variable = None
    
    for v in candidate_variables:
        auc_v = auc(current_variables + [v], target, basetable_train_f)
        
        if auc_v >= best_auc:
            best_auc = auc_v
            best_variable = v
   
    return best_variable

#Printing the next best variable to include
next_variable = next_best(current_variables, candidate_variables, target, basetable_train_f)
print(next_variable)

current_variables = []
candidate_variables = ["Capital_Region","gender_F", "gender_M", "language_N", "language_F","amount5yrs", "min_amount5yrs","max_amount5yrs","mean_amount5yrs"]

#Here we calculate the best 5 variables to include in the predictive model
max_number_variables = 5
number_iterations = min(max_number_variables, len(candidate_variables))
for i in range(0,number_iterations):
    next_variable = next_best(current_variables,candidate_variables,target,basetable_train_f)
    current_variables = current_variables + [next_variable]
    candidate_variables.remove(next_variable)
print(current_variables)

############################Building Machine Learning Models
#https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
np.any(np.isnan(basetable_train_f))
pd.isnull(basetable_train_f).sum() > 0
pd.isnull(basetable_test_f).sum() > 0
del basetable_train_f['zipcode']
del basetable_test_f['zipcode']
X_train = basetable_train_f.drop("donated", axis=1)
Y_train = basetable_train_f["donated"]
X_test  = basetable_test_f.drop("donorID", axis=1).copy()

#Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

#Linear Support Vector Machine
from sklearn.svm import SVC, LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)

#BEST MODEL
results = pd.DataFrame({
    'Model': ["Random_Forst","Logistic_Regression","Support_Vector_Machine","Decision_Tree"],
    'Score': [acc_random_forest,acc_log,acc_linear_svc,acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
#########################################################################################
#Cumulative gains in Python
#http://eric.univ-lyon2.fr/~ricco/cours/slides/PJ%20-%20en%20-%20machine%20learning%20avec%20scikit-learn.pdf

import scikitplot as skplt 
import matplotlib.pyplot as plt
skplt.metrics.plot_cumulative_gain(y, predictions)
plt.show()

#lift curve in python
skplt.metrics.plot_lift_curve(y, predictions)
plt.show()

#Business case using lift curve
population_size = 25000
target_incidence = 0.03
reward_target = 35
cost_campaign = 0.5
def profit(perc_targets, perc_selected, population_size,reward_target, cost_campaign):
    cost = cost_campaign * perc_selected * population_size
    reward = reward_target * perc_targets * perc_selected * population_size
    return(reward - cost)
perc_selected = 0.40
lift = 1.2
perc_targets = lift * target_incidence

print(profit(perc_targets, perc_selected, population_size,
            reward_target, cost_campaign))