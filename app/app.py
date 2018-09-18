import nbgwas
from nbgwas import Nbgwas

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)


class nbgwasapp(Resource): 
    def get(self, jobid):
        #TODO Check of jobid exists 
        return f"Checked {jobid}!"

    def post(self, jobid): 
        args = reqparse.parse_args()
        print(args)

# api.add_resource(HelloWorld, '/')
api.add_resource(nbgwasapp, '/nbgwas/<jobid>', endpoint='nbgwas')

if __name__ == '__main__':
    app.run(debug=True) #Change this in production