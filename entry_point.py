import argparse
import os
import datetime as dt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import joblib
from io import StringIO

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

def predict_fn(input_object, model):
    y_pred = model.predict_proba(input_object)[0][1]
    return y_pred

def input_fn(request_body, request_content_type):
    print(request_body)
   
    df = pd.read_csv(StringIO(request_body), header=None)
   
    print(df)
   
    return df.to_numpy()

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    args, _ = parser.parse_known_args()
    train = pd.read_csv('{}/train.csv'.format(args.train))
    test = pd.read_csv('{}/test.csv'.format(args.test))
    
    
    
    train['fecha'] = pd.to_datetime(train['fecha'])
    train['fecha'] = train['fecha'].map(dt.datetime.toordinal)

    test['fecha'] = pd.to_datetime(test['fecha'])
    test['fecha'] = test['fecha'].map(dt.datetime.toordinal)
    
    
    X_train = np.array(train['fecha']).reshape(-1, 1)
    y_train = np.array(train['value']).reshape(-1, 1)

    X_test = np.array(test['fecha']).reshape(-1, 1)
    y_test = np.array(test['value']).reshape(-1, 1)
    
    model = LinearRegression()

    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))
    
    
    y_pred = model.predict(X_test)

    print(f"Coeficientes del modelo: {model.coef_}")
    print(f"Intresección del modelo: {model.intercept_}")
    print(f"Número de coeficientes del modelo: {len(model.coef_)}")
    
    puntuation = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Score r2: {puntuation}")
    print(f"Score mae: {mae}")
    print(f"Score mse: {mse}")

    
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib') )
    
    print(train)
