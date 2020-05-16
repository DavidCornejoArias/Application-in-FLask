#.\chatbotWebApp\Scripts\activate.ps1, initialize environemnt
# Reading from packages
from flask import Flask, render_template, request, render_template_string, redirect, jsonify
import simplejson as json
from flask_ngrok import run_with_ngrok
import ast
from flask_sqlalchemy import SQLAlchemy
from bot import Bot
from main import modelsProcess, testingPreprocessing, chatting
import pandas as pd
from os import environ
# page access token for facebook
page_access_token = ''
# Uploading a model
try:
    loadedmodelFirst = modelsProcess('json',"h5")
    loadedmodel = loadedmodelFirst.modelPreprocessing()
    testing = testingPreprocessing("json")
    words, labels, intents, doc_x, doc_y = testing.processing()
except:
        # preparing testing
    testing = testingPreprocessing("json")
    # Creating training and output data
    words, labels, intents, doc_x, doc_y = testing.processing()
    training, output = testing.trainingData(doc_x,doc_y,words,labels)
    # training the model
    testing.training(training, output, 'json', 'h5')
    # loading the trained model
    loadedmodelFirst = modelsProcess('json',"h5")
    loadedmodel = loadedmodelFirst.modelPreprocessing()
# Initializing app
app = Flask(__name__)
run_with_ngrok(app)

# Connecting to our database
class Config(object):
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test.db?check_same_thread=False'

app.config.from_object(Config)
db = SQLAlchemy(app)
db.init_app(app)
db.create_all()
db.echo = True
connection = db.engine.connect()
columnsNames = ['tag','patterns','responses','context_set']
testJsonAdress = 'intenciones.json'
# Function for transforming this particular Json type to Pandas DataFrame
def transformingJsonToPandasDataFrame(addressJSON,columns,firstLayer):
    with open(addressJSON,encoding='utf-8') as file:
        data = json.load(file,encoding='utf-8')
        df = pd.DataFrame()
    for column in columns:
        columnList= []
        for i in range(len(data[firstLayer])):
            columnList.append(data[firstLayer][i][column])
        df[column]=columnList
    return df
def transformingPandasJson(df,columnsList,columnsString,groupName,saveName):
    listOfDic = []
    totalDic = {}
    for row in range(len(df[df.columns[0]])):
        columnDic = {}
        context = columnsString[-1]
        columnString = columnsString[:-1]
        for column in columnString:
            columnDicAdd =  {str(column):list(df[column])[row]}
            columnDic.update(columnDicAdd)
        for column in columnsList:
            columnDicAdd =  {str(column):ast.literal_eval(list(df[column])[row])}
            columnDic.update(columnDicAdd)
        # adding context
        columnDicAdd =  {str(context):list(df[context])[row]}
        columnDic.update(columnDicAdd)
        listOfDic.append(columnDic)
    totalDic[groupName] = listOfDic
    with open(saveName,'w+', encoding='utf-8') as f:
        json.dump(totalDic, f, ensure_ascii=False, indent=4)
# Getting from json, a pandas dataframe
df = transformingJsonToPandasDataFrame(testJsonAdress,columnsNames,'intents')
# changing columns to string
df['tag']= df['tag'].astype('str')
df['patterns']= df['patterns'].astype('str')
df['responses']= df['responses'].astype('str')
df['context_set']=df['context_set'].astype('str')
df.to_sql('clientHistory', con=connection, if_exists='replace')

#df['context_set']
#df.columns
# Creating the model
class Intents(db.Model):
    __tablename__ = "clientHistory"
    __table_args__ = {'extend_existing': False} 
    tag = db.Column(db.String(200), primary_key=True)
    patterns = db.Column(db.String(20000), nullable = False)
    responses = db.Column(db.String(20000), nullable = False)
    context_set = db.Column(db.String(20000), nullable = True)
    def __repr__(self):
        return '<Intention %r>' % self.tag
# Creating an index page
@app.route('/', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        if token == 'secret':
            return str(challenge)
        return render_template('index.html')
    elif request.method == 'POST':
        data = json.loads(request.data)
        messaging_events = data['entry'][0]['messaging']
        bot = Bot(page_access_token)
        for message in messaging_events:
            user_id = message['sender']['id']
            text_input = message['message'].get('text')
            print('Message from user ID {} - {}'.format(user_id, text_input))

            bot.send_text_message(user_id, chatting(text_input,words).chat(labels,loadedmodel, intents))
    
        return render_template('index.html')
@app.route('/data/',methods=['POST','GET'])
def data():
    if request.method == "POST":
        tagContent = request.form['tag']
        patternContent = request.form['pattern']
        responseContent = request.form['response']
        context_set = request.form['context_set']
        newtag = Intents(tag= tagContent,patterns=patternContent,responses=responseContent,context_set=context_set)
        try:
            db.session.add(newtag)
            db.session.commit()
            return redirect('/data/')
        except:
            return 'There was an issue adding your task'
    else:
        #stmt = 'select * from clientHistory'
        #result_proxy = connection.execute(stmt)
        #results = result_proxy.fetchall()
        tags = Intents.query.all()
        return render_template('record.html', tags=tags,columns=columnsNames)
# delete a certain tag in the dataframe
@app.route('/data/delete/<tag>')
def delete(tag):
    tag_to_delete = Intents.query.get_or_404(tag)
    try:
        db.session.delete(tag_to_delete)
        db.session.commit()
        return redirect('/data/')
    except:
        return "There was a problem deleting that task"

@app.route('/data/save/', methods=['POST','GET'])
def save():
    if request.method == "POST":
#        try:
        df = pd.read_sql_query("SELECT * FROM clientHistory", connection,index_col=None)
        columns = ['patterns','responses']
        columnString = ['tag','context_set']
        # transforming from Pandas to Json
        transformingPandasJson(df,columns,columnString,'intents','json')
        # preparing testing
        testing = testingPreprocessing("json")
        # Creating training and output data
        words, labels, intents, doc_x, doc_y = testing.processing()
        training, output = testing.trainingData(doc_x,doc_y,words,labels)
        # training the model
        testing.training(training, output, 'json', 'h5')
        # loading the trained model
        loadedmodelFirst = modelsProcess('json',"h5")
        loadedmodel = loadedmodelFirst.modelPreprocessing()
        testing = testingPreprocessing("json")
        words, labels, intents, doc_x, doc_y = testing.processing()
        return redirect('/')
    else:
        return "Not working"
#        except:
#            return "There was a problem saving the dataset"

if __name__ == '__main__':
    app.run()
