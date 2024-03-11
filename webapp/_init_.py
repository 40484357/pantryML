from flask import Flask, url_for
from flask_sqlalchemy import SQLAlchemy
from os import path

def create_app():
    application = Flask(__name__)
    application.config['SECRET_KEY'] = 'flavortown'

    from .views import views
    
    application.register_blueprint(views, url_prefix='/')

    return application