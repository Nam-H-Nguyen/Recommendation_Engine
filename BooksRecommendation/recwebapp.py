from sklearn.decomposition import TruncatedSVD
import sklearn
import IPython.display as Disp
import requests
from numpy import int64
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import os

# Path of HTML templates
TEMPLATE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'templates')
# Start Flask App with HTML/CSS/JS
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)

# Default local host is port 5000
@app.route("/")
def hello():
    return TEMPLATE_DIR


# Verbose update on what happens when the python application is launched
print("Building Recommendation Engine . . .")

# Read in data sets using Panda library
print("Reading data . . .")
books_table = pd.read_csv("./dataset/books.csv")
ratings_table = pd.read_csv("./dataset/ratings.csv", encoding='UTF-8',
                            dtype={'user_id': int, 'book_id': int, 'rating': int})
print("Finished reading in books and ratings datasets")

# Merge the books and ratings datasets based on book_id
print("Merging datasets . . .")
books_table_2 = books_table[['book_id', 'original_title', 'average_rating', 'books_count',
                             'original_publication_year', 'authors', 'image_url']]
combined_books_table = pd.merge(ratings_table, books_table, on='book_id')
print("Finished merging datasets")

# Create a three dimensional pivot table of row: users, column: book titles and the ratings as the values
print("Creating Pivot table of users, book titles, and ratings . . .")
pt_matrix = combined_books_table.pivot_table(
    values='rating', index='user_id', columns='original_title', fill_value=0)
# Transpose the row with the columns for SVD
tranX_matrix = pt_matrix.values.T
print("Finished transposing matrices")

# Compress column with TruncatedSVD to make calculation of coefficient manageable
print("Beginning SVD . . .")
SVD = TruncatedSVD(n_components=20, random_state=17)
svd_matrix = SVD.fit_transform(tranX_matrix)
print("Generated SVD Matrix")

# Use Numpy library to calculate the Pearson coefficient for similarity of books based on categories
print("Building similarity coefficient matrix . . .")
corr_matrix = np.corrcoef(svd_matrix)
print("Finished calculating Pearson coefficients for book similarities")

# Get the book names in a list
print("Getting the list of book names in the matrix")
book_names = pt_matrix.columns
book_list = list(book_names)

print("Finish building the recommendation engine")


def getRecommendations(bookName):
    # Find the index of where the book name is in the correlation matrix
    book_name_index = book_list.index(bookName)
    corr_book = corr_matrix[book_name_index]

    # Filter books from 0.75 to 1.0 Pearson Coefficient in the correlation matrix
    rec_list = list(book_names[(corr_book < 1.0) & (corr_book > 0.75)])

    # max = 5
    # if(len(rec_list) < 5):
    #     max = len(rec_list)
    return books_table_2[books_table_2.original_title.isin(rec_list)]

# Start with rec.html with GET method, then POST method when a query exists
@app.route("/rec", methods=['GET', 'POST'])
def rec():
    query = ''
    if(request.method == "POST"):
                # Get the User query from the form text field
        query = request.form.get('query')
        recommendations = getRecommendations(query)

        # Render the html template with the query value with the user query and fire up the recommendations algorithm
        return render_template('rec.html', query=query, recommendations=recommendations.to_html())
    else:
                # User has not issued a query yet
        return render_template('rec.html', query="", recommendations="<<unknown>>")


# Flask App run on localhost:5000/rec
if __name__ == "__main__":
    app.run(use_reloader=False)
