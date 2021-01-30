# Copyright by Riccardo Rinaldi
#
# Markupsafe mitigates injection attacks, meaning untrusted user input 
# can safely be displayed on a page.
#
# Use url_for(<function_name>, arg1, arg2, ...) to dynamically build a URL
#

from flask import Flask, request, url_for, render_template, redirect, session, flash
from markupsafe import escape
from werkzeug.utils import secure_filename
from fog05 import FIMAPI
from fog05_sdk.interfaces.FDU import FDU
import json
import os

# UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'json'}

app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'8\xfb\xe6b\x84=\x7fl\xf0m\xcc\x92\xc1\x1f:\xcew\xdbq\x0eY\xe3\xdb\xcf'


@app.route('/')
def index():
    if 'yaksIp' in session:
        return 'you are connected to %s' % escape(session['yaksIp'])
    return redirect(url_for('tutorial'))
    
    
@app.route('/css')
def css():
    url = url_for('static', filename='style.css')
    return redirect(url)
   
    
@app.route('/dashboard/')
@app.route('/dashboard/<yaks_ip>')
def dashboard(yaks_ip='127.0.0.1'):
    session['yaksIp'] = escape(yaks_ip)
    # TODO: fog05 stuff goes here
    global a
    a = FIMAPI(session['yaksIp'])
    nodes = a.node.list()
    if len(nodes) == 0:
        app.logger.error('No nodes at {}'.format(session['yaksIp']))
        return redirect(url_for('close'))

    session['nodes'] = nodes
    session['instances'] = dict()
    app.logger.debug('Nodes: {}'.format(nodes))
    return render_template('dashboard.html')
    
    
@app.route('/tutorial')
def tutorial():
    return 'The tutorial is coming soon'


@app.route('/close')
def close():
    session.clear()
    return redirect(url_for('index'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            app.logger.error('No file in POST request')
            return redirect(request.referrer)
        file = request.files['file']
        if file.filename == '':
            app.logger.error('No selected file')
            return redirect(request.referrer)
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            try:
                fdu_d = FDU(json.loads(file.read()))
                session['fdu_descriptor'] = fdu_d.to_json()
                app.logger.debug('File upload succeeded')
                global a
                res = a.fdu.onboard(fdu_d)
                app.logger.debug(res.to_json())
                e_uuid = res.get_uuid()
                session['fdu_uuid'] = e_uuid
            except ValueError:
                app.logger.error("Fdu configuration json is malformed")
            return redirect(request.referrer)
        else:
            app.logger.error('Extension not allowed')
            return redirect(request.referrer)


@app.route('/deploy/<node>')
def deploy(node=''):
    if node == '':
        app.logger.error('Node id missing in deploy request')
        return redirect(request.referrer)
    app.logger.debug('Define fdu {} at {}'.format(session['fdu_uuid'], node))
    inst_info = a.fdu.define(session['fdu_uuid'], node)
    instance_id = inst_info.get_uuid()
    app.logger.debug('Created new instance with id: {}'.format(instance_id))
    a.fdu.configure(instance_id)
    app.logger.debug('Congratulations! You deployed the fdu. Ready to start')
    session['instances'][node] = instance_id
    # app.logger.debug(session['instances'])
    return redirect(request.referrer)


@app.route('/start/<node>')
def start(node=''):
    app.logger.debug('Start here the fdu at {}'.format(node))
    return redirect(request.referrer)
