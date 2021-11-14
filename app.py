from flask import Flask
from flask_restful import Resource, Api
from main import Recommend

import logging

app = Flask(__name__)
api = Api(app)


class Test(Resource):
    def get(self, path):
        return {'is_direction': True, 'no_use_def': ['first', 'second'], 'corr': 70, 'path': path}


api.add_resource(Recommend, '/post', methods=['POST','GET'])

if __name__ == '__main__':
    app.run(debug=True, port=81)
