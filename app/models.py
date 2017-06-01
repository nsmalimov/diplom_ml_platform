# -*- coding: utf-8 -*-

# TODO not only one data in project
# TODO migration
# TODO а можно ли в Project хранить id связанных компонентов или спрашивать каждый раз из view?
import datetime

from sqlalchemy.types import DateTime

from app.util.funcs import ml_path
from run import db

epoch = datetime.datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


def dump_datetime(value):
    """Deserialize datetime object into string form for JSON workers."""
    if value is None:
        return None
    return value.strftime('%Y-%m-%dT%H:%M:%S')


class Project(db.Model):
    __tablename__ = "project"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), unique=True, nullable=False)
    description = db.Column(db.String(120), default="None")
    modified_at = db.Column(DateTime, default=datetime.datetime.now, nullable=False)

    records = db.relationship("Data", backref="project", lazy='dynamic',
                              primaryjoin="Project.id == Data.project_id")
    algorithms = db.relationship("Algorithm", backref="project", lazy='dynamic',
                                 primaryjoin="Project.id == Algorithm.project_id")
    analys_classif = db.relationship("AnalysClassif", backref="project", lazy='dynamic',
                                     primaryjoin="Project.id == AnalysClassif.project_id")

    @property
    def serialize(self):
        """Return object data in easily serializeable format"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'modified_at': dump_datetime(self.modified_at),
            # This is an example how to deal with Many2Many relations
            # 'analys_classif': self.analys_classif.serialize_many2many
        }

    # @property
    # def serialize_many2many(self):
    #    return [item.serialize for item in self.many2many]

    def __init__(self, title, description):
        self.title = title
        self.description = description

    def __repr__(self):
        return '<Project %r>' % self.id


class Data(db.Model):
    __tablename__ = "data"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(80), nullable=False)

    # classification clustering universal
    task_type = db.Column(db.String(80), nullable=False)

    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True, default=None)

    @property
    def serialize(self):
        """Return object data in easily serializeable format"""
        return {
            'id': self.id,
            'filename': self.filename,
            'project_id': self.project_id,
            'task_type': self.task_type
        }

    def __init__(self, filename, task_type):
        self.filename = filename
        self.task_type = task_type

    def __repr__(self):
        return '<Data %r>' % self.id


class Algorithm(db.Model):
    __tablename__ = "algorithm"

    #TODO
    # уникальность может повторяться
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), unique=False, nullable=False)
    description = db.Column(db.String(120), default="None")
    filename = db.Column(db.String(80), unique=False)
    preloaded = db.Column(db.Boolean, default=False, nullable=False)
    type = db.Column(db.String(80))

    # keras fe
    custom_save_model = db.Column(db.Boolean, default=False, nullable=False)

    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True, default=None)

    @property
    def serialize(self):
        """Return object data in easily serializeable format"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'preloaded': self.preloaded,
            'project_id': self.project_id,
            'filename': self.filename,
            'type': self.type,
            'custom_save_model': self.custom_save_model
        }

    def __init__(self, title, description, preloaded, filename, type, custom_save_model):
        self.title = title
        self.description = description
        self.preloaded = preloaded
        self.filename = filename
        self.type = type
        self.custom_save_model = custom_save_model

    def __repr__(self):
        return '<Algorithm %r>' % self.id


class ResultTypes(db.Model):
    __tablename__ = "result_types"
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(80), unique=True, nullable=False)
    title = db.Column(db.String(80), unique=False, nullable=False)

    @property
    def serialize(self):
        return {
            'id': self.id,
            'title': self.title,
            'name': self.name
        }

    def __init__(self, name, title):
        self.name = name
        self.title = title

    def __repr__(self):
        return '<ResultTypes %r>' % self.id


concat_many_to_many = db.Table('concater_table',
                               db.Column('algorithm_id', db.Integer, db.ForeignKey('algorithm.id')),
                               db.Column('analys_classif_id', db.Integer, db.ForeignKey('analys_classif.id')),
                               db.Column('result_types_id', db.Integer, db.ForeignKey('result_types.id'))
                               )


class AnalysClassif(db.Model):
    __tablename__ = "analys_classif"
    id = db.Column(db.Integer, primary_key=True)

    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True, default=None)

    record = db.Column(db.Integer, db.ForeignKey('data.id'), nullable=True, default=None)

    algorithms = db.relationship('Algorithm', secondary=concat_many_to_many,
                                 backref=db.backref('algorithm', lazy='dynamic'))

    result_types = db.relationship('ResultTypes', secondary=concat_many_to_many,
                                   backref=db.backref('result_types', lazy='dynamic'))

    @property
    def serialize(self):
        """Return object data in easily serializeable format"""
        return {
            'id': self.id,
            'project_id': self.project_id,
            'analys': self.analys
        }

    def __init__(self, record):
        self.record = record

    def __repr__(self):
        return '<Analys and classification %r>' % self.id


db.create_all()


def insert_common_algs_to_db():
    path = ml_path + "common_algorithms"
    alg_names_classification = ["log_reg", "random_forest", "svm"]

    alg_names_clustering = ["k_means", "mini_batch_kmeans", "birch"]

    # TODO
    # если не scikit

    for i in alg_names_classification:
        res = Algorithm.query.filter_by(title=i).first()
        type = "classification"
        if (res == None):
            algorithm = Algorithm(i, "scikit " + i, True, path + "/" + type + "/" + i + ".py", type, False)
            db.session.add(algorithm)

    for i in alg_names_clustering:
        res = Algorithm.query.filter_by(title=i).first()
        type = "clustering"
        if (res == None):
            algorithm = Algorithm(i, "scikit " + i, True, path + "/" + type + "/" + i + ".py", type, False)
            db.session.add(algorithm)

    db.session.commit()


def insert_result_types_to_db():
    arr = [
        # TODO другие типы
        ("automaticle_best_model", "Найти лучшую модель и вывести метрики"),
        # ("classify", "Классифицировать"),
        # ("metrics", "Метрики")

        ("train_save_metrics_graphics", "Обучить, получить метрики и графики"),
    ]

    for i in arr:
        res = ResultTypes.query.filter_by(name=i[0]).first()
        if (res == None):
            resultType = ResultTypes(i[0], i[1])
            db.session.add(resultType)

    db.session.commit()

insert_common_algs_to_db()
insert_result_types_to_db()
