# Imports
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer

# Data
train = pd.read_csv('train_merge.csv')
test = pd.read_csv('test_merge.csv')

# Preprocessing
train_ids = train['SK_ID_CURR']
labels = train['TARGET']
train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test_ids = test['SK_ID_CURR']
test = test.drop(columns = ['SK_ID_CURR'])


categorical_features_1 = []
categorical_features_2 = []
for col in train.select_dtypes(['object', 'bool']):
    if len(list(train[col].unique())) <= 2:
        categorical_features_1.append(col)
    else:
        categorical_features_2.append(col)

numerical_features = train.select_dtypes(['float64', 'int64']).columns.tolist()

numerical_pipeline = make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler(feature_range=(0, 1)))
categorical_pipeline_1 = make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'), OrdinalEncoder())
categorical_pipeline_2 = make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'), OneHotEncoder(handle_unknown='ignore'))

preprocessor = make_column_transformer((numerical_pipeline, numerical_features), (categorical_pipeline_1, categorical_features_1), (categorical_pipeline_2, categorical_features_2))

def get_column_names_from_ColumnTransformer(column_transformer):
    """
    Helper function which explores a Column Transformer to get feature names
    Parameters
    ----------
    column_transformer: a sklearn column_transformer
    Returns
    ----------
    a list of strings of the column names of the outputs of the colun_transformer
    """
    col_name = []
    for transformer_in_columns in column_transformer.transformers_:
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        if isinstance(transformer, OneHotEncoder):
            if isinstance(raw_col_name, str):
                names = transformer.get_feature_names(input_features=[raw_col_name])
            else:
                names = transformer.get_feature_names(input_features=raw_col_name)
        elif isinstance(transformer, PolynomialFeatures):
            names = transformer.get_feature_names(input_features=raw_col_name)
        else:
            try:
                names = transformer.get_feature_names()
            except AttributeError:
                # if no 'get_feature_names' function, use raw column name
                names = raw_col_name
        if isinstance(names, np.ndarray):  # eg.
            col_name += names.tolist()
        elif isinstance(names, list):
            col_name += names
        elif isinstance(names, str):
            col_name.append(names)
    return col_name

train = preprocessor.fit_transform(train)
train = pd.DataFrame(train, columns=get_column_names_from_ColumnTransformer(preprocessor))

train['TARGET'] = labels
train['SK_ID_CURR'] = train_ids

test = preprocessor.transform(test)
test = pd.DataFrame(test, columns=get_column_names_from_ColumnTransformer(preprocessor))

test['SK_ID_CURR'] = test_ids

train.to_csv('train_prep.csv', index=False)
test.to_csv('test_prep.csv', index=False)