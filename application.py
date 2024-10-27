from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template("index.html")
    else:
        data=CustomData(gender=request.form.get('gender'),race=request.form.get('race'),parent_education=request.form.get('parent_education'),lunch=request.form.get('lunch'),test_course=request.form.get('test_course'),writing_score=request.form.get('writing_score'),reading_score=request.form.get('reading_score'))
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('index.html',results=results[0])
if __name__ =='__main__':
    app.run() 
    