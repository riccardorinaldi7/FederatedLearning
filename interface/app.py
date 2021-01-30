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

plugins = {'892ae9fe-9d1b-4dc6-87a3-66f5695a9971': 'LXD',
           'b5947df7-ff6e-4b72-827b-a6f5b2be6b70': 'Docker',
           '809839f2-b511-489d-a8cf-8c80d714893e': 'native',
           '8fb33188-846c-45f8-83df-2a25e6b78049': 'KVM'}


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
def dashboard(yaks_ip=''):
    if yaks_ip == '':
        if 'yaksIp' in session:
            return render_template('dashboard.html')
        else:
            return render_template('connect.html')
    else:
        session['yaksIp'] = escape(yaks_ip)
        # fog05 initialization goes here
        global a
        a = FIMAPI(session['yaksIp'])
        nodes = a.node.list()
        if len(nodes) == 0:
            app.logger.error('No nodes at {}'.format(session['yaksIp']))
            return redirect(url_for('close'))

        session['nodes'] = nodes
        # session['instances'] = dict()
        app.logger.debug('Nodes: {}'.format(nodes))
        return render_template('dashboard.html')

    
@app.route('/connect')
def connect():
    return render_template('connect.html')


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
            return redirect(url_for('dashboard'))
        file = request.files['file']
        if file.filename == '':
            app.logger.error('No selected file')
            return redirect(url_for('dashboard'))
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            try:
                fdu_d = FDU(json.loads(file.read()))
                session['fdu_descriptor'] = fdu_d.to_json()
                app.logger.debug('File upload succeeded')
                res = a.fdu.onboard(fdu_d)
                app.logger.debug(res.to_json())
                e_uuid = res.get_uuid()
                session['fdu_uuid'] = e_uuid
            except ValueError:
                app.logger.error("Fdu configuration json is malformed")
            return redirect(url_for('dashboard'))
        else:
            app.logger.error('Extension not allowed')
            return redirect(url_for('dashboard'))


@app.route('/deploy/<node>')
def deploy(node=''):
    if node == '':
        app.logger.error('Node id missing in deploy request')
        return redirect(url_for('dashboard'))
    app.logger.debug('Define fdu {} at {}'.format(session['fdu_uuid'], node))
    inst_info = a.fdu.define(session['fdu_uuid'], node)
    instance_id = inst_info.get_uuid()
    app.logger.debug('Created new instance with id: {}'.format(instance_id))
    a.fdu.configure(instance_id)
    app.logger.debug('Congratulations! You deployed the fdu. Ready to start')
    session['instance/{}'.format(node)] = instance_id
    session[instance_id] = 'READY'
    # app.logger.debug(session['instance'])
    return redirect(url_for('dashboard'))


@app.route('/start/<node>')
def start(node=''):
    if node != '':
        instance_id = session['instance/{}'.format(node)]
        a.fdu.start(instance_id)
        app.logger.debug('Started fdu at {}'.format(node))
        session[instance_id] = 'STARTED'
    return redirect(url_for('dashboard'))


@app.route('/stop/<node>')
def stop(node=''):
    if node != '':
        instance_id = session['instance/{}'.format(node)]
        a.fdu.stop(instance_id)
        app.logger.debug('Stopped fdu at {}'.format(node))
        session[instance_id] = 'STOPPED'
    return redirect(url_for('dashboard'))


@app.route('/remove/<node>')
def remove(node=''):
    if node != '':
        instance_id = session['instance/{}'.format(node)]
        a.fdu.clean(instance_id)
        a.fdu.undefine(instance_id)
        app.logger.debug('Removed fdu from {}'.format(node))
        session.pop(instance_id)  # remove the instance state
        session.pop('instance/{}'.format(node))  # remove the instance from instances
        app.logger.debug('Session updated')
    return redirect(url_for('dashboard'))


@app.route('/offload')
def offload():
    a.fdu.offload(session['fdu_uuid'])
    session.pop('fdu_descriptor')
    session.pop('fdu_uuid')
    app.logger.debug('FDU configuration discarded')
    return redirect(url_for('dashboard'))


@app.route('/migrate')
def migrate():
    mig_node = escape(request.args.get('mig_node', ''))
    if mig_node == '':
        app.logger.error('Migration node id not within get request')
    else:
        app.logger.debug('Migrating FDU to {}'.format(mig_node))
    return redirect(url_for('dashboard'))
