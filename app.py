from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>hello duniya</h1>"


if __name__ == '__main__':
        app.run()
