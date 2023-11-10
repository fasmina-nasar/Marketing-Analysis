Term Deposit Marketing Analysis
==============================

Poject Overview
------------

Thie aim of this project is to predict if a customer will subscribe to a term deposit or not leveraging information coming from call center data. Conduct data analysis to find ways to improve the success rate for calls made to customers for any product that clients offers. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for clients to make informed decisions. 
Dataset consists of 13 attributes(consists of both categorical and numerical) and a target column(categorical).

    age : age of customer (numeric)

    job : type of job (categorical)

    marital : marital status (categorical)

    education (categorical)

    default: has credit in default? (binary)

    balance: average yearly balance, in euros (numeric)

    housing: has a housing loan? (binary)

    loan: has personal loan? (binary)

    contact: contact communication type (categorical)

    day: last contact day of the month (numeric)

    month: last contact month of year (categorical)

    duration: last contact duration, in seconds (numeric)

    campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

    Output (desired target): y - has the client subscribed to a term deposit? (binary)

Exploratory Data Analysis
------------
 Based on the Data Analysis performed,  we've come up with these conclusions:

 * Outliers are present in Age, Balance, Duration and Campaign, Among which Balance have maximum outliers of around 11%.
 * Dataset is highly imbalanced - negative dominates positive by 92.7%.
 * According to the chi-square test and visualizations, "default" doesn't seem like statistically significant in determinig whether clent subscribes to term deposit or not.

Models Implemented
------------

I've used 5 fold cross validation for both the results of over sampling and under sampling data to compare and choose best. I've compiled Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, KNeighbors Classifier, Linear SVC, Extra trees Classifier, Decision Tree Classifier along with parameters. 

Among these Random Forest Classifier and Extra tree classifier are having accuracy of 95% by conducting Over sampling.

Conclusions
------------

* Most of the clients are either blue-collar or management, then comes technician.
* In marital status, single and divorced accepts the offer comparatively to Married ones. Eventhough count of Married people for accpeting offer is high, their rejection rate is also high.
* People who has credit in default and housing loan are more likely to decline the offer.
* Contacting people in the month of April and in the interval june to august will be of positive result.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
