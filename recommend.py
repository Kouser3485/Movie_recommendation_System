from distutils.command import clean #used to imprt clean which cleans the data
import numpy as np #numpy to perform mathematical operations
import pandas as pd #padas which just acts as excel in our python
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer #these libraries are imported to convert the data into vectors
from sklearn.metrics.pairwise import cosine_similarity #evaluate the vecotors to find the most similar vectors
import nltk # NLP kit 
import re #regulating expression library
# nltk.download('stopwords') =if stopwords not downloaded you can download using this command
from nltk.corpus import stopwords #importing the  stopwords from nltk kit
import string



#first read the csv file.
data = pd.read_csv("C:/Users/DELL/Desktop/netflix recomend/netflixData.csv") 

#there will be lot of columns in any file ,select the prefferd columns that you want to work on!
data = data[["Title", "Content Type", "Description", "Genres"]]

#then check if any null values are there using data.isnull(),if yes then droup out the null values doing as below
data = data.dropna()

#create a set of stopwords imported from ntlk kit,pupose is creating a set is:in sets the same values are not repeated twice
stopword = set(stopwords.words('english'))#english represents that everything being done is only on english words



#declare and define a function that cleans the inputed dataset
def cleaner(text):
    text = str(text).lower()#data converted to lower
    text = re.sub('\[.*?\]', '', text)# here [.*?] ,https?://S+/www.S+ etc ..these are replaced by an empty string in text paseed
    text = re.sub('https?://\S+/www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    return " ".join(text) #returns the cleaned text that is joined to function call again

#here you call an function cleaner ,this function is applied on dataset of title
data["Title"] = data["Title"].apply(cleaner)




# Combine the text features (Title and Genres) for TF-IDF transformation
combined_text = data["Title"] + " " + data["Genres"]

# Use CountVectorizer to create the count matrix and then apply TfidfTransformer to convert text into vectors
count_vectorizer = CountVectorizer(stop_words='english')
X_count = count_vectorizer.fit_transform(combined_text)
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(X_count)

#instance of cosine similarity is created ,and it find the most similar vectors in the inputs passed
similarity = cosine_similarity(tfidf_matrix)

#`indices` is a Pandas Series mapping unique movie titles to their corresponding indices in the DataFrame `data`.
indices = pd.Series(data.index, index=data["Title"]).drop_duplicates()





#now after a good cleaning,create a function to recoomand movies corresponding to inputed movie:

def recommendation(title):
    # Convert the title to lowercase for case-insensitive matching
    if title not in indices:
        print(f"Movie with title '{title}' not found in the our platform.")
        return None

    index = indices[title]#gives the index of title that stored in index
    similar_scores = list(enumerate(similarity[index]))#find similar vectors that matches the vectors of movie title
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)#sorting occurs  to find the most similar vecotors, like the vector at bottom with highest matching ,is considered as most similar to passed title
    similar_scores = similar_scores[1:11]  # Exclude the movie itself,gives top 11 similar movies from sorted one
    movie_indices = [i[0] for i in similar_scores]# extract the indexs from similarity list
    
    print("YOU MAY ALSO LIKE THESE MOVIES WHICH ARE SAME AS {title} \n")
    recommended_movies = data["Title"].iloc[movie_indices].tolist()# corresponding movies matching to indexed are put up in a list
    for movie_title in recommended_movies:
        print(movie_title)

    return recommended_movies


    
print("Hello!,welocome to movie recommendation ,enter you favourite movie: ")
title=input()
print(recommendation(title))
#print(indices["unwell"])# check if function is working!!


#the data here is converted to vectors because:when we convert data into vectors,all the data lies on the same plane,then it becomes easy for consine similarity function to match up the similar vectors that corresponds to similar data 











