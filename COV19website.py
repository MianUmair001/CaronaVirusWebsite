from flask import Flask, escape, request,render_template,url_for,flash,redirect
from GettingLiveData import getData
app = Flask(__name__)

data=getData()
print("data yah h ",data)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html',data=data)


@app.route('/about')
def about():
    return render_template('about.html',title='About')







if __name__ == "__main__":
        app.run(debug=True) 