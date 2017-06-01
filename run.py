from flask import Flask

from werkzeug.contrib.fixers import ProxyFix

app = Flask(__name__)

# TODO
# если базы данных нет, то удалить все проекты

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

UPLOAD_FOLDER = '/files/'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

app.config.from_object('instance.config.DevelopConfig')

db = SQLAlchemy(app)

migrate = Migrate(app, db)

db.create_all()

db.session.commit()

from app.views import *

app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run()
