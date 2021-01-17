from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# load the model from disk
model = pickle.load(open('model.pkl' , 'rb'))
cv = pickle.load(open('transformer.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('Food Reviews Sentiment.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':

		message = request.form['message']

		#make submitted text into  a list
		#sample_review = [message]

		#lower case text
		sample_review = message.lower()

		#remove punctuation
		sample_review = ''.join([char for char in sample_review if char not in punctuation])

		#strip any leading/trailing whitespaces
		sample_review = sample_review.strip()

		#substitute multiple spaces with a single space
		sample_review = re.sub('\s+', " ", sample_review)   

		#split/tokenize reviews
		split_review = sample_review.split()

		#remove stopwords:
		split_review = [word for word in split_review if not word in set(stopwords.words('english'))]

		#stemming words:
		stemming = PorterStemmer()
		review = [stemming.stem(word) for word in split_review]

		#Joing to create a single string for each review 
		review = ' '.join(review)

		#make into a list
		review = [review]

		#vertorizing the input review
		input_review = cv.transform(review).toarray()


		prediction = model.predict(input_review)

		if prediction == 1:
			prediction = "Positive Review"
		else:
			prediction = "Negative Review"


		return render_template('Food Reviews Sentiment Result.html',prediction_text = "Predicted Sentiment: {}".format(prediction))

	else:
		return render_template('Food Reviews Sentiment.html')

if __name__ == '__main__':
	app.run(debug=True)





