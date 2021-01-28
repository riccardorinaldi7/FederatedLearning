#!/bin/bash

echo "Debug mode: on. Serving on any IP..."

export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0
