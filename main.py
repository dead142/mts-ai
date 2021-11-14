from flask import Flask
from flask_restful import Resource, Api
from flask import request
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy
import json
import pickle
import catboost
import sqlite3

app = Flask(__name__)
api = Api(app)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Recommend(Resource):
    def __init__(self):
        sqlite_connection = sqlite3.connect('identifier.sqlite')
        self.cursor = sqlite_connection.cursor()

    def recommendation(self, cluster):
        print(cluster)
        cluster = int(cluster)
        print(type(cluster))
        params = {}
        if cluster == 0:
            params = {"min_price": 100, "max_price": 100}
        if cluster == 1:
            params = {"min_price": 1000, "max_price": 9999}
            print(11)
        if cluster == 2:
            params = {"min_price": 10000, "max_price": 10000000}
        print(params)
        print(params['min_price'])
        sqlite_select_query = """SELECT  *  from items   WHERE
    price > ? AND     price < ?"""
        self.cursor.execute(sqlite_select_query, [params['min_price'],params['max_price']])
        rows = self.cursor.fetchall()
        return rows
        # найти историю выбрать категорию товаров
        #  предложить рандомный товар
        # Если купит, то welcome

    def post(self):
        request_data = request.get_json()
        df = pd.json_normalize(request_data)
        # df = pd.read_json("json_ex.json", orient="records")
        model = pickle.load(open("models/pay_class.sav", 'rb'))
        y_pred = model.predict(df)
        rows = self.recommendation(y_pred[0])
        return json.dumps(rows)

    def __preprocess(self, df, boolType, objectType):
        le = LabelEncoder()
        mappings = {}
        for i in boolType:
            df[i] = df[i].astype(int)
        for i in objectType:
            df[i] = df[i].fillna('0')
            le.fit(df[i])
            # le.classes_ = numpy.load("models/le/"+ i+".npy")
            df[i] = le.transform(df[i])
            # df[i] = df.apply(lambda i:le.transform(df[i].astype(str)), axis=0, result_type='expand')
            mappings[i] = dict(zip(le.classes_, le.transform(le.classes_)))
        return mappings
