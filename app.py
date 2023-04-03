from flask import Flask,render_template,request
from text_summary import summarizer

app = Flask(__name__)
# Serve static files
app.static_folder = 'static'
app.static_url_path = '/static'

@app.route('/')
def index():
    return render_template('index8.html')

@app.route('/analyse',methods=['GET','POST'])
def analyse():
    if request.method=='POST':
        rawtext=request.form['rawtext']
        summary, original_txt,len_ori_txt,len_summary, kwords= summarizer(rawtext)
    return render_template('summary2.html',summary=summary,original_txt=original_txt,len_ori_txt=len_ori_txt,len_summary=len_summary,kwords=kwords)



if __name__=="__main__":
    app.run(debug=False,host='0.0.0.0')

