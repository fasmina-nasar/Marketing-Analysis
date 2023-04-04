from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline



class Model_creation:

  def create_model(self, X_train, y_train, X_test, y_test, classifier, params):
    """
    creates an ML model for classification and perform K-Fold cross - validation and grid search to find the best hyperparameters.

    parameters: 
    X_train : training input features
    y_train : training target variable
    X_test : testing input features
    y_test : testing target variable
    classifier : ML classifier to use
    params : hyperparameters

    returns:
    mean accuracy score across all folds
    """

    # Create the pipeline
    pipeline = Pipeline([('classifier' , classifier)])

    # Create the KFold object
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X_train):
      X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
      y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=kf, verbose=1, n_jobs=-1)

    # Fit the GridSearchCV object
    grid_search.fit(X_train_fold, y_train_fold)

     # Get the best estimator
    best_estimator = grid_search.best_estimator_

    # Predict on the validation set
    y_pred = best_estimator.predict(X_val_fold)

    # Calculate the accuracy score, f1 score,roc auc score
    print(' accuracy score: ',accuracy_score(y_val_fold, y_pred))
    print('f1 score: ',f1_score(y_val_fold,y_pred))
    print('roc_auc: ',roc_auc_score(y_val_fold,y_pred))
    
    #check best parameters
    best_params = grid_search.best_params_
    print(best_params)
    return y_pred