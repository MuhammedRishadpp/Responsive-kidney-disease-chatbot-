from flask import Flask, render_template, request
from newspaper import Article
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Article converted to text
article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/diagnosis-treatment/drc-20354527')
article.download()
article.parse()
article.nlp()
corpus = article.text

# Tokenization
sentence_list = nltk.sent_tokenize(corpus)

# A Function to return a random greeting response to a user's greeting
def greeting_response(text):
    text = text.lower()
    bot_greetings = ['howdy', 'hi', 'hello']
    user_greetings = ['hi', 'hey', 'hello', 'hola', 'greeting']
    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)

def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))
    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                # swap
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index

# Create the bot's response
def bot_response(user_input):
    user_input = user_input.lower()
    sentence_list.append(user_input)
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(cm[-1], cm)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    response_flag = 0
    j = 0
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            bot_response = bot_response + '' + sentence_list[index[i]]
            response_flag = 1
            j += 1
        if j > 2:
            break
    if response_flag == 0:
        bot_response = bot_response + '' + "I apologize, I don't understand."
    sentence_list.remove(user_input)
    return bot_response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    if greeting_response(user_input):
        return {'response': greeting_response(user_input)}
    else:
        return {'response': bot_response(user_input)}

if __name__ == '__main__':
    app.run(debug=True)