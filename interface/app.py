# Copyright by Riccardo Rinaldi
#
# Markupsafe mitigates injection attacks, meaning untrusted user input 
# can safely be displayed on a page.
#
# Use url_for(<function_name>, arg1, arg2, ...) to dynamically build a URL
#

from flask import Flask, request, url_for, render_template, redirect, session, flash
from markupsafe import escape
from fog05 import FIMAPI
from fog05_sdk.interfaces.FDU import FDU
import json

app = Flask(__name__)

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
    a = FIMAPI(session['yaksIp'])
    nodes = a.node.list()
    if len(nodes) == 0:
        app.logger.error('No nodes at {}'.format(session['yaksIp']))
        return redirect(url_for('close'))

    app.logger.info('Nodes: {}'.format(nodes))
    return render_template('dashboard.html', node_list = nodes)
    
    
@app.route('/tutorial')
def tutorial():
    return 'The tutorial is coming soon'
    
@app.route('/close')
def close():
    session.clear()
    return redirect(url_for('index'))
