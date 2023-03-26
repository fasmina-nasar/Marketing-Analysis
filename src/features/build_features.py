class preparing_data:
    def encode_features(df):
      for col in df.columns:
        if col in categorical_features:
          unique_values = df[col].unique()
          mapping = {value:index for index,value in enumerate(unique_values)}
          df[col] = df[col].map(mapping)
      X = df.drop('y',axis = 1)
      y = df.y
      return X,y


    def preprocess_data(features,target):
      """ 
          Removed outliers 
          split features and target into training and testing part
          Balance dataset by applying SMOTE
          scale features
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

      # Applying SMOTE (Synthetic Minority Over-Sampling Techniques)
      print('before smote :', Counter(y_train))
      smote = SMOTE()
      X_train,y_train = smote.fit_resample(X_train , y_train )
      print('after smote :', Counter(y_train))

      # Scaling features
      scaler=StandardScaler()
      X_train=scaler.fit_transform(X_train)
      X_test=scaler.transform(X_test)

      return X_train,X_test,y_train,y_test