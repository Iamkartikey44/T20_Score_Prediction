from flask import Flask,render_template,request,jsonify,url_for
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

pipe = pickle.load(open('pipe.pkl','rb'))
#Top teams 
teams = ['Australia','India','England','New Zealand','Bangladesh','Sri Lanka','South Africa','Pakistan','Afghanistan','West Indies',
          'Namibia','Ireland','Netherlands','Zimbabwe','Scotland','United Arab Emirates']

cities = ['Dubai','Abu Dhabi','Colombo','Mirpur','Johannesburg','London','Harare','Hamilton','Auckland','Cape Town','Al Amarat','Pallekele','Kuala Lumpur',
 'Barbados','Chittagong','Sharjah','Dublin','Durban','Sydney','Melbourne','Singapore','St Lucia','Nagpur','Lahore','Wellington','Nottingham',
 'Lauderhill','Centurion','Manchester','Kampala','Castel','Bangkok','Mumbai','Dhaka','Southampton','Sylhet','Edinburgh','Mount Maunganui',
 'Kolkata','Belfast','Hambantota','Greater Noida','Delhi','Trinidad','Windhoek','Guyana','Murcia','Chandigarh','Adelaide',
 'Port Moresby','Bangalore','St Kitts','Rotterdam','Cardiff','Christchurch','St Peter Port']

app = Flask(__name__)




@app.route('/')
def home():
    return render_template('index.html',venue=cities)


@app.route('/example/<score>/')
def example(score):
    return render_template("result.html", score=score)


@app.route('/predict',methods=['POST'])

def predict():
            
    batting_team = request.json['batting']
    bowling_team = request.json['bowling']
    city = request.json['venues']
    

    overs = float(request.json['over'])
    runs = int(request.json['run'])
    wickets = int(request.json['wicket'])
    runs_in_prev_5 = int(request.json['run_last5'])
    balls_left = 120 - (overs*6)
    wickets_left  = 10 - wickets
    curr_rr =  runs/overs

    input_df = pd.DataFrame(
        {'batting_team':[batting_team],'bowling_team':[bowling_team],'current_score':[runs],'wicket_left':[wickets_left],'curr_rr':[curr_rr],
        'city':[city],'ball_left':[balls_left],'last_five':[runs_in_prev_5]}
    )
    result =  pipe.predict(input_df)
    my_prediction = str(int(result[0]))
   



    
    

    return jsonify({'redirect': url_for("example",score = my_prediction)})

        


if __name__ == '__main__':
    app.run(debug=True,port=8000)
