import os

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)


@app.route('/')
def index():
   return "please hit the endpoint /hello"


@app.route('/hello', methods=['GET'])
def hello():
   # name = request.form.get('name')
    return "hello Sanjay"



if __name__ == '__main__':
   app.run()
