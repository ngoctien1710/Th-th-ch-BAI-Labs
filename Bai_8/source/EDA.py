import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
def EDA_train(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_col = ['id', 'rate', 'spkts', 'dpkts', 'swin', 
                'sbytes', 'dloss', 'is_sm_ips_ports', 'ct_srv_dst',
                'is_ftp_login', 'ct_srv_src', 'ct_src_dport_ltm', 'tcprtt', 'attack_cat',]
    df = df.drop(columns=drop_col, axis=1)
    X = df.drop(columns=['label'], axis=1)
    y = df['label']
    
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns
    scaler = MinMaxScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    X = pd.get_dummies(X, drop_first=True)
    smote = SMOTE(random_state = 42)
    X, y = smote.fit_resample(X, y)
    return X, y, scaler

def EDA_test(df_test: pd.DataFrame, train_cols: list, scaler) -> tuple[pd.DataFrame, pd.Series]:
    #scaler = MinMaxScaler()
    drop_col = ['id', 'rate', 'spkts', 'dpkts', 'swin', 
                'sbytes', 'dloss', 'is_sm_ips_ports', 'ct_srv_dst',
                'is_ftp_login', 'ct_srv_src', 'ct_src_dport_ltm', 'tcprtt', 'attack_cat',]
    df_test = df_test.drop(columns=drop_col, axis=1)
    X_test = df_test.drop(columns=['label'], axis=1)
    y_test = df_test['label']
    
    numeric_cols = X_test.select_dtypes(include=['int64','float64']).columns
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    for col in train_cols:
        if col not in X_test:
            X_test[col] = 0
    X_test = X_test[train_cols]
    
    return X_test, y_test