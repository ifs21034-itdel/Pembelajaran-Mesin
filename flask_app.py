from flask import Flask, render_template, request, url_for, redirect # import the Flask class
#We also import the render_template, request, url_for, and redirect functions
app = Flask(__name__) # create a variable called app that is an instance of the Flask class, passing in the module name
app.secret_key = b'\xa3\x14\xa1B]\x8a\xda\xd3\xbf\xbf\x03E{\x1aYx'


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score

df=pd.read_csv('df_log_reg.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.drop(['Test_score'], axis=1, inplace=True)
df2=pd.read_csv('input.csv')
df2.drop(['Unnamed: 0'],axis=1,inplace=True)
df2.drop(['Test_score'], axis=1, inplace=True)

X = df.drop(['Admit'],axis=1)
y = df.Admit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logmodel = LogisticRegression(random_state=42)
logmodel.fit(X_train,y_train)
y_pred_test=logmodel.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test,y_pred_test)


univers = ['Arizona State University', 'Boston University',
       'Carnegie Mellon University', 'University of Delaware',
       'Drexel University', 'University of Arizona',
       'Illinois Institute of Technology', 'Iowa State University',
       'Indiana University Bloomington',
       'University of Minnesota, Twin Cities', 'Northeastern University',
       'Northwestern University', 'New York University',
       'Pennsylvania State University',
       'Rochester Institute of Technology',
       'Rensselaer Polytechnic Institute',
       'Rutgers University-New Brunswick', 'Rutgers University, Newark',
       'Santa Clara University', 'Stevens Institute of Technology',
       'University at Buffalo SUNY', 'Syracuse University',
       'Texas A&M; University, College Station',
       'University of California, Irvine', 'University of Cincinnati',
       'University of Florida', 'University of Illinois at Chicago',
       'University of Maryland, College Park',
       'University of North Carolina at Charlotte',
       'University of Texas at Dallas',
       'University of Texas at Arlington',
       'University of Texas at Austin', 'University of Utah',
       'University of Washington', 'Worcester Polytechnic Institute']

courses = ['Management Information System',
       'Information Management and Systems', 'Information Systems',
       'Information Science', 'ICS with concentration in Informatics',
       'Information Technology and Management', 'Information Management']

# Routes tell the Flask app where to go when we are typing URLs into the browser
# We can list two functions for the route
@app.route('/')
@app.route('/home/')
def home(): # create a function for the route
       return render_template('home.html', uni=univers, cour=courses) # we are returning the text, "Hello, World!"
		
@app.route('/results/', methods=["GET","POST"])
def results():

       if request.method == "POST":
              year = float(request.form.get('year'))
              work = float(request.form.get('work'))
              gre = float(request.form.get('gre'))
              uni_score = float(request.form.get('undergrad'))
              uni_score = uni_score/100 
              univ = request.form.get('uni')
              course = request.form.get('cour')
              df2['GRE_SCORE'] = gre
              df2['Undergrad_score'] = uni_score
              df2['Grad_Year'] = year
              df2['Work_Month'] = work
              df2[univ] = 1
              df2[course] = 1
              result_entry = logmodel.predict(df2)
              if result_entry == 1:
                     return render_template('congrats.html')
              else:
                     return render_template('results.html')
       else:
              return redirect(url_for('home'))

@app.route('/disclaimer/')
def disclaimer():
       return render_template('disclaimer.html')

if __name__ == "__main__":
       app.run(debug=True)
