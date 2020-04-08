from flask import Flask, escape, request,render_template,url_for,flash,redirect
from GettingLiveData import getData
app = Flask(__name__)

data=getData()
print("data yah h ",data)

@app.route('carona-virus-website.herokuapp.com')
@app.route('carona-virus-website.herokuapp.com/home')
def home():
    return render_template('index.html',data=data)


if __name__ == "__main__":
        app.run() 