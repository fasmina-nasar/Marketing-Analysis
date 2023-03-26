class numerical_and_categorical:
  def __init__(self,df):
    self.df = df
  
  # visualizing numerical features
  def num_feats(self):
    plt.figure(figsize=(8,10))
    count=1
    while count<=len(numerical_features):
      for i in numerical_features:
        plt.subplot(3,2,count)
        sns.boxplot(x=self.df[i])
        count +=1
        plt.title('visualizing {}'.format(i))
      plt.tight_layout()
      plt.show()

  # visualizing categorical features
  def cat_feats(self):
    plt.figure(figsize=(12,14))
    i=1
    while i < len(categorical_features):
      for col in categorical_features[:-1]:
        plt.subplot(4,2,i)
        sns.histplot(data=self.df , x=col, hue=categorical_features[-1],kde=True)
        i+=1
      plt.tight_layout()
      plt.show()    


class freqs_and_countplot:
  def __init__(self, df, col):
    self.df = df
    self.col = col
  def count_freqs(self):
    print(pd.crosstab(self.df[self.col],self.df['y']).T)

  def countplot(self):
    sns.countplot(x=self.df[self.col],hue=self.df['y'])
    plt.xticks(rotation=45, size=12)
    plt.xlabel(self.col)
    plt.ylabel("count")
    plt.title(f"{self.col} and Subscriptions")
    plt.show()

