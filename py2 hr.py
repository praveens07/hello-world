import pandas as pd
import numpy as np

hr_df = pd.read_csv('F:HR1.csv')
hr_df.head()
hr_df.columns

#Encoding Categorical Features
numerical_features = ['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours', 'time_spend_company']
categorical_features = ['Work_accident','promotion_last_5years', 'sales', 'salary']

#An utility function to create dummy variable
#def create_dummies( df, colname ):
def create_dummies( df, colname ) :    #run it with full syntax to avoid eof
    col_dummies = pd.get_dummies(df[colname], prefix=colname)
    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df

for c_feature in categorical_features:#run for fully
  hr_df = create_dummies( hr_df, c_feature )
hr_df.head()  

# Validating and Splitting the dataset
feature_columns = hr_df.columns.difference( ['left'] )
feature_columns

from sklearn.cross_validation import train_test_split


train_X, test_X, train_y, test_y = train_test_split( hr_df[feature_columns],hr_df['left'],test_size = 0.2,random_state = 42 )

hr_left_df = pd.DataFrame( hr_df.left.value_counts() )
hr_left_df

import matplotlib.pyplot as plt
import seaborn as sn
sn.barplot( hr_left_df.index, hr_left_df.left )

#Building Models
#Logistic Regression Model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit( train_X, train_y )

list( zip( feature_columns, logreg.coef_[0] ) )

logreg.intercept_
#Predicting the test cases
hr_test_pred = pd.DataFrame( { 'actual':  test_y,'predicted': logreg.predict( test_X ) } )
hr_test_pred = hr_test_pred.reset_index()

#Comparing the predictions with actual test data
hr_test_pred.sample( n = 10 )

#Creating a confusion matrix
from sklearn import metrics

cm = metrics.confusion_matrix( hr_test_pred.actual,hr_test_pred.predicted, [1,0] )
cm

import seaborn as sn
sn.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Left", "No Left"] , yticklabels = ["Left", "No Left"] )
plt.ylabel('True label')
plt.xlabel('Predicted label')#run 3 lines

score = metrics.accuracy_score( hr_test_pred.actual, hr_test_pred.predicted )
round( float(score), 2 )# 78%


#Building Decision Tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.grid_search import GridSearchCV
param_grid = {'max_depth': np.arange(3, 10)}

tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 10)
tree.fit( train_X, train_y )
tree.best_params_
tree.best_score_#98%

#Build Final Decision Tree Model
clf_tree = DecisionTreeClassifier( max_depth = 9 )
clf_tree.fit( train_X, train_y, )

tree_test_pred = pd.DataFrame( { 'actual':  test_y,'predicted': clf_tree.predict( test_X ) } )
    
tree_test_pred.sample( n = 10 )
metrics.accuracy_score( tree_test_pred.actual, tree_test_pred.predicted )
tree_cm = metrics.confusion_matrix( tree_test_pred.predicted,tree_test_pred.actual,[1,0] )

sn.heatmap(tree_cm, annot=True,fmt='.2f',xticklabels = ["Left", "No Left"] , yticklabels = ["Left", "No Left"] )
plt.ylabel('True label')
plt.xlabel('Predicted label')#run 3 liines

#Generate rules from the decision tree
export_graphviz( clf_tree,out_file = "hr_tree.odt",feature_names = train_X.columns )
import pydotplus as pdot# install conda install -c conda-forge pydotplus,conda install -c conda-forge/label/gcc7 pydotplus  
chd_tree_graph = pdot.graphviz.graph_from_dot_file( 'hr_tree.odt' )
chd_tree_graph.write_jpg( 'hr_tree.jpg' )#conda install -c anaconda graphviz 
from IPython.display import Image
Image(filename='hr_tree.jpg')

#Random Forest Model
from sklearn.ensemble import RandomForestClassifier
radm_clf = RandomForestClassifier()
radm_clf.fit( train_X, train_y )

radm_test_pred = pd.DataFrame( { 'actual':  test_y,'predicted': radm_clf.predict( test_X ) } )
metrics.accuracy_score( radm_test_pred.actual, radm_test_pred.predicted )

tree_cm = metrics.confusion_matrix( radm_test_pred.predicted,radm_test_pred.actual,[1,0] )

sn.heatmap(tree_cm, annot=True,fmt='.2f',xticklabels = ["Left", "No Left"] , yticklabels = ["Left", "No Left"] )
plt.ylabel('True label')
plt.xlabel('Predicted label')#run 3 lines

#Feature Importance from Random Forest Model    
indices = np.argsort(radm_clf.feature_importances_)[::-1]
feature_rank = pd.DataFrame( columns = ['rank', 'feature', 'importance'] )
for f in range(train_X.shape[1]):# run fully for
  feature_rank.loc[f] = [f+1,
                         train_X.columns[indices[f]],
                         radm_clf.feature_importances_[indices[f]]]
sn.barplot( y = 'feature', x = 'importance', data = feature_rank )
