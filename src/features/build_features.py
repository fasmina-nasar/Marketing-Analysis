from collections import Counter
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.filterwarnings("ignore")


class preparing_dataset:
  def encode_features(self,df):
    for col in df.columns:
      if col in categorical_features:
        unique_values = df[col].unique()
        mapping = {value:index for index,value in enumerate(unique_values)}
        df[col] = df[col].map(mapping)
    X = df.drop(['y','default','day'],axis = 1)
    y = df.y
    return X,y

  def preprocess_data(self,features,target):
    """ 
        Removed outliers 
        split features and target into training and testing part
        Balance dataset by conducting both oversampling (using SMOTE) and under sampling(using NearMiss)
        scale features(using standard scaler)
    """

    # outlier removal
    for col in features.columns:
      if col in numerical_features:
        iqr = (features[col].quantile(0.75)) - (features[col].quantile(0.25))
        lower_limit = (features[col].quantile(0.25)) - 1.5* iqr
        upper_limit = (features[col].quantile(0.75)) + 1.5* iqr
        features = features[~ ((features[col]<lower_limit) | (features[col]>upper_limit))]
    target = target[features.index]

    # split features and target into training and testing part
    X_train,X_test,y_train,y_test=train_test_split(features , target , test_size=0.25,random_state=10)

    # Applying SMOTE to conduct over sampling (Synthetic Minority Over-Sampling Techniques)
    print('Over sampling results: ')
    print('before smote :', Counter(y_train))
    smote = SMOTE()
    X_train_sm,y_train_sm = smote.fit_resample(X_train , y_train )
    print('after smote :', Counter(y_train_sm))

    # Applying NearMiss to conduct under sampling
    print('Under sampling results: ')
    print('before NearMiss :',Counter(y_train))
    nm=NearMiss()
    X_train_nm, y_train_nm = nm.fit_resample(X_train,y_train)
    print('after NearMiss :',Counter(y_train_nm))

    # Scaling features
    scaler=StandardScaler()
    X_train_nm=scaler.fit_transform(X_train_nm)
    X_train_sm=scaler.fit_transform(X_train_sm)
    X_test=scaler.transform(X_test)

    return (X_train_nm, X_train_sm, X_test, y_train_nm, y_train_sm, y_test)