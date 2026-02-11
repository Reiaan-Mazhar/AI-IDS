
import numpy as np 
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
import os

def unique_values(df, columns):
        for column_name in columns:
            unique_vals = df[column_name].unique()
            value_counts = df[column_name].value_counts()
            print(f"Unique Values ({len(unique_vals)}): {unique_vals}\n")
            print(f"Value Counts:\n{value_counts}\n{'='*40}\n")


def nskdd_preprocess(path_nslkdd='/home/ntenna_ech/fyp/data_downloaders/nslkdd/',kdd_train_path="/home/ntenna_ech/fyp/data_downloaders/nslkdd/KDDTrain+.txt"):
    for dirname, _, filenames in os.walk(path_nslkdd):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    df_0 = pd.read_csv(kdd_train_path)
    df= df_0.copy()
    df.head()

    columns = (['duration'
    ,'protocol_type'
    ,'service'
    ,'flag'
    ,'src_bytes'
    ,'dst_bytes'
    ,'land'
    ,'wrong_fragment'
    ,'urgent'
    ,'hot'
    ,'num_failed_logins'
    ,'logged_in'
    ,'num_compromised'
    ,'root_shell'
    ,'su_attempted'
    ,'num_root'
    ,'num_file_creations'
    ,'num_shells'
    ,'num_access_files'
    ,'num_outbound_cmds'
    ,'is_host_login'
    ,'is_guest_login'
    ,'count'
    ,'srv_count'
    ,'serror_rate'
    ,'srv_serror_rate'
    ,'rerror_rate'
    ,'srv_rerror_rate'
    ,'same_srv_rate'
    ,'diff_srv_rate'
    ,'srv_diff_host_rate'
    ,'dst_host_count'
    ,'dst_host_srv_count'
    ,'dst_host_same_srv_rate'
    ,'dst_host_diff_srv_rate'
    ,'dst_host_same_src_port_rate'
    ,'dst_host_srv_diff_host_rate'
    ,'dst_host_serror_rate'
    ,'dst_host_srv_serror_rate'
    ,'dst_host_rerror_rate'
    ,'dst_host_srv_rerror_rate'
    ,'attack'
    ,'level'])

    df.columns = columns
    df.isnull().sum()
    cat_features = df.select_dtypes(include='object').columns
    unique_values(df, cat_features)
    df.duplicated().sum()
    
    attack_n = []
    for i in df.attack :
        if i == 'normal':
            attack_n.append("normal")
        else:
            attack_n.append("attack")
    df['attack'] = attack_n 
    df["protocol_type"].value_counts(normalize=True)
    cat_features = df.select_dtypes(include='object').columns
    cat_features

    le=preprocessing.LabelEncoder()
    clm=['protocol_type', 'service', 'flag', 'attack']
    
    for x in clm:
        df[x]=le.fit_transform(df[x])

    X = df.drop(["attack"], axis=1)
    y = df["attack"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=43) 

    train_index = X_train.columns
    mutual_info = mutual_info_classif(X_train, y_train)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = train_index
    mutual_info.sort_values(ascending=False)
    Select_features = SelectKBest(mutual_info_classif, k=30)
    Select_features.fit(X_train, y_train)
    train_index[Select_features.get_support()]

    columns=['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'wrong_fragment', 'hot', 'logged_in', 'num_compromised',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate']


    X_train=X_train[columns]
    X_test=X_test[columns]
    return X_train,X_test,y_train,y_test
