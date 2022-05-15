from sklearn.model_selection import train_test_split
from flask import Flask, flash, jsonify, session, render_template, url_for, request, redirect, abort
import pandas as pd
import numpy as np
import os
from flask_sqlalchemy import SQLAlchemy
from nltk.stem.porter import PorterStemmer
from matplotlib import pyplot as plt
import pymysql
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_admin import BaseView, expose
from flask_admin.form import rules
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from flask_mysqldb import MySQL
import MySQLdb.cursors
from sqlalchemy import create_engine


app = Flask(__name__)
app.secret_key = 'esmeraldabloodfreeze'
# database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost/data-edom'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///edom.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://ymeixwlzmrdzqt:bfbb322ed081b4bd3fc8dbfc7aaf6e5706311b413525f0a76e58faf5d27821de@ec2-44-196-223-128.compute-1.amazonaws.com:5432/d6v8etpsv0kql7'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
admin = Admin(app, name='Klasifikasi Emosi', template_mode='bootstrap4')


class formedom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    semester = db.Column(db.String(50))
    dosen = db.Column(db.String(100))
    mk = db.Column(db.String(100))
    message = db.Column(db.Text)
    predict = db.Column(db.Integer)


class UserView(ModelView):
    can_delete = False  # disable model deletion
    can_export = True


class SecureModelView(UserView):
    def is_accessible(self):
        if "logged_in" in session:
            return True
        else:
            abort(403)


class Rekap(BaseView):
    @expose('/')
    def index(self):
        results = formedom.query.with_entities(formedom.predict).all()
        df = pd.DataFrame(results)
        df.columns = ['predict']
        data = df.groupby('predict').size().reset_index(name='jumlah')
        dat = pd.DataFrame(data.jumlah)
        my_labels = 'Kurang', 'Cukup', 'Baik', 'Sangat Baik'
        my_color = ['red', 'aqua', 'green', 'blue']
        dat.plot(kind='pie', labels=my_labels, autopct='%1.1f%%',
                 colors=my_color, subplots=True, stacked=True, legend=False)
        # plt.rcParams["figure.figsize"] = [7.00, 3.50]
        # plt.rcParams["figure.autolayout"] = True
        # df.groupby('predict').size().reset_index(name='jumlah').plot(kind="bar", color=[
        #     'red', 'aqua', 'green', 'blue'], x='predict', y='jumlah', stacked=True, legend=False, ylim=(0, 25))
        plt.title('Hasil Seluruh Sentimen')
        plt.xlabel('Sentimen')
        plt.ylabel("")
        plt.savefig('static/img/image.png')
        # Rekap = submit()
        return self.render('admin/rekap.html')


class RekapDosen(BaseView):
    @expose('/')
    def index(self):
        RekapDosen = submit()
        return self.render('admin/rekapdosen.html')


class Logout(BaseView):
    @expose('/')
    def index(self):
        session.clear()
        return redirect('/')


admin.add_view(SecureModelView(formedom, db.session))
admin.add_view(Rekap(name='Rekap', endpoint='Rekap'))
admin.add_view(RekapDosen(name='RekapDosen', endpoint='RekapDosen'))
admin.add_view(Logout(name='Logout', endpoint='Logout'))

# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'data-edom'

# mysql = MySQL(app)

# Definitions


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100


# Modeling
data = pd.read_csv("datx.csv")
data.columns = ["Kritik dan Saran", "Klasifikasi"]
# Features and Labels
data['label'] = data['Klasifikasi']
# .map(
#     {'Kurang': 0, 'Cukup': 1, 'Baik': 2, 'Sangat Baik': 3})
data['tidy_tweet'] = np.vectorize(remove_pattern)(
    data['Kritik dan Saran'], "@[\w]*")
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet
data['body_len'] = data['Kritik dan Saran'].apply(
    lambda x: len(x) - x.count(" "))
data['punct%'] = data['Kritik dan Saran'].apply(lambda x: count_punct(x))
X = data['tidy_tweet']
y = data['label']
# Extract Feature With CountVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(X)  # Fit the Data
X = pd.concat([data['body_len'], data['punct%'],
               pd.DataFrame(X.toarray())], axis=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Using Classifier
clf = svm.SVC(C=1, gamma=1, kernel='linear')
clf.fit(X, y)

semester_matakuliah = {
    'Semester 1': ['Kalkulus', 'Perangkat Keras Komputer', 'Aplikasi Perkantoran', "Bahasa Inggris 1", 'Pendidikan Agama', "Logika Informatika", 'Algoritma dan Struktur Data'],
    'Semester 2': ["Kalkulus 2", "Analisa dan Perancangan Sistem", "Organisasi & Arsitektur Komputer", "Sistem Operasi", "Algorutma dan Struktur Data 2", "Sistem Digital", "Bahasa Inggris 2", "Kewarganegaraan"],
    'Semester 3': ["Pemrograman Komputer", "Jaringan Komputer", "Pemrograman Web", "Sistem Basis Data", "Matematika Numerik", "Statistika", "Enterprise Resource Planning", "Desain Grafus dan Multimedia"],
    'Semester 4': ["Pemrograman Komputer 2", "Jaringan Komputer 2", "Pemrograman Web 2", "Sistem Basis Data 2", "Sistem Pendukung Keputusan", "Data Mining", "Pengantar Kecerdasan Buatan", "Pengabdian Pada Masyarakat", "Kunjungan Industri"],
    'Semester 5': ["Mobile Programming", "Komputasi dan Aplikasi Cloud", "Framework Programming", "Data Warehouse", "Pengujian Perangkat Lunak", "Machine Learning", "Pemrograman Sistem Cerdas", "Pengolahan Citra Digital", "Leadership"],
    'Semester 6': ["Metodologi Penelitian", "Rekayasa Perangkat Lunak", "Big Data", "Mobile Development", "Game & Design Development"],
    'Semester 7': ["Sistem Informasi Manajemen", "Soft Skill", "Manajemen Projek IT", "Kerja Praktik Industri", "Kuliah Kerja Lapangan"],
    'Semester 8': ["Technopreneurship", "Etika Profesi IT"]
}


@app.route('/')
def home():
    return render_template('home.html')


# @app.route('/mk')
# def mk():
#     semester = request.args.get('semester')
#     list_of_mk = semester_matakuliah[semester]
#     return render_template('mk_option.html', list_of_mk=list_of_mk)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['password'] == 'admin' and request.form['username'] == 'admin':

            session['logged_in'] = True
            return redirect('/admin')
        else:
            return render_template('login.html')

    else:
        return render_template('login.html')

    # if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
    #     username = request.form['username']
    #     password = request.form['password']
    #     cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    #     cursor.execute('SELECT * FROM accounts WHERE username = % s AND password = % s', (username, password, ))
    #     account = cursor.fetchone()
    #     if account:
    #         session['loggedin'] = True
    #         session['id'] = account['id']
    #         session['username'] = account['username']
    #         return render_template('index.html')
    #     else:
    #         msg = 'Incorrect username / password !'
    # return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


@app.route('/edom')
def edom():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        semester = request.form['semester']
        mk = request.form['mk']
        dosen = request.form['dosen']
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len, punct, vect], axis=1)
        my_prediction = clf.predict(total_data)
        prediction = my_prediction
    # cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # cursor.execute(
    #     'INSERT INTO edom VALUES (NULL, % s,% s , % s, % s, % s)', (semester, dosen, mk, message, int(predict)))
    # mysql.connection.commit()
    predict = formedom(semester=semester, dosen=dosen, mk=mk,
                       message=message, predict=int(prediction))

    db.session.add(predict)
    db.session.commit()

    return render_template('index.html', prediction=my_prediction)


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        query = request.form['dosen']
        dosen = formedom.query.with_entities(
            formedom.predict).filter_by(dosen=query).all()
        data = pd.DataFrame(dosen)
        data.columns = ['predict']
        df = data.groupby('predict').size().reset_index(name='jumlah')
        high = df['jumlah'].max()
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        df.plot(kind="bar", color=[
            'red', 'aqua', 'green', 'blue'], x='predict', y='jumlah', stacked=True, legend=False, ylim=(0, (high+5)))
        plt.title('Hasil Seluruh Sentimen Dosen ' + query)
        plt.xlabel('Sentimen')
        plt.savefig('static/img/rekapdosen.png')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=4000)
