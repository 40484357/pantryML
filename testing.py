import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances
import csv


I = pd.read_csv('recipes/interactions_train.csv')
R = pd.read_csv('recipes/RAW_recipes.csv')

I.info()
R.info()

I.rating.value_counts().plot(kind = 'bar', fontsize = 14, 
                             figsize = (5, 2)).set_title('Distribution of Rating',
                                                         fontsize=16, ha = 'center', va = 'bottom')

#plt.show() #shows a plot graph of the distribution of ratings

_all = I.drop(['date','u','i'], axis = 1) 
_all

# Data clean up #

#research showed that the file can be too big and cause memory issues, it was recommended to reduce the file and data looked at. I reduced it to the top 10k users who give the most reviews, and 10k recipes that received the most reviews.
#To do this I have grouped 'recipe_id' and the 'reviews_count' and 'user_id' and 'reviews_count' and created an aggregate for most reviews per ID -- I then sort them to get the top 10k. 
grouped_recipes = _all.groupby(['user_id'], as_index = False, sort = False).agg({'recipe_id':'count'}).reset_index(drop = True)
grouped_recipes = grouped_recipes.rename(columns = {'recipe_id':'reviews_count'})
grouped_recipes = grouped_recipes.sort_values('reviews_count', ascending = False).iloc[:10000, :]


grouped_users = _all.groupby(['recipe_id'], as_index = False, sort = False).agg({'user_id':'count'}).reset_index(drop = True)
grouped_users = grouped_users.rename(columns = {'user_id':'reviews_count'})
grouped_users = grouped_users.sort_values('reviews_count', ascending = False).iloc[:10000, :]
#I then merge the grouped_recipes with grouped_users and _all to get a new data frame this gets me a total of 231636 qualified rows.
_part = pd.merge(_all.merge(grouped_recipes).drop(['reviews_count'], axis = 1), grouped_users).drop(['reviews_count'], axis = 1)
_part



conn = sqlite3.connect('data.db')



_part.to_sql('data', conn, if_exists='replace', index=False)



"""
userIds = dict(zip(list(_part['user_id'].unique()),
                   list(range(len(_part['user_id'].unique())))))

highestID = max(userIds, key = userIds.get)
newUserID = str(highestID + 1)

print('new user ID', newUserID)

conn.execute(
             create table part_data as
             select * from data
            )


print('highestID', highestID)
print('newID', newUserID)

df2 = {'user_id' : newUserID, 'recipe_id' : 0.0, 'rating' : 0}

_part = _part.append(df2, ignore_index = True)

try:
    cursor = conn.cursor()

    sqlite_insert_query = INSERT INTO part_data (user_id, recipe_id, rating) VALUES (?, ?, ?)

    data_tuple = (newUserID, 0, 0.0)
    cursor.execute(sqlite_insert_query, data_tuple)
    conn.commit()
    print('successfully added')
    cursor.close()
except sqlite3.Error as error:
    print('failed to insert', error)

"""



sql_query = pd.read_sql_query('''SELECT
                              * FROM part_data
                              ''', conn)



_part = pd.DataFrame(sql_query, columns = ['user_id', 'recipe_id', 'rating'])



grouped_user = _part.groupby(['user_id'], as_index = False, sort = False).agg({'recipe_id':'count'}).reset_index(drop = True)
grouped_user = grouped_user.rename(columns = {'recipe_id':'reviews_count'})

grouped_recipe = _part.groupby(['recipe_id'], as_index = False, sort = False).agg({'user_id':'count'}).reset_index(drop = True)
grouped_recipe = grouped_recipe.rename(columns = {'user_id':'reviews_count'})

#I found from 'grouped_user' that the mean amount of reviews by users is 26.72 and the mean amount of reviews a recipe received is 26.61 (both rounded).
#print(grouped_user[['reviews_count']].describe())
#print(grouped_recipe[['reviews_count']].describe())

#from I i ran a distribution of rating, I did it again in _part this was to make sure that the distribution of ratings wasn't changed significantly when reducing the number of data samples from the original sample.
#the results show a similar distribution 
_part.rating.value_counts().plot(kind = 'bar', fontsize = 14, 
                                 figsize = (5, 2)).set_title('Distribution of Rating',
                                                             fontsize = 16, ha = 'center', va = 'bottom')

#plt.show()

#recipe and user IDs would fall out of range when mapping values in the training data sets, to solve this a guide showed to assign new IDs to the users and recipes that were qualified above.
#clean_df contains the reworked IDs ('clean_dataframe')
set_userID = dict(zip(list(_part['user_id'].unique()),
                      list(range(len(_part['user_id'].unique())))))

#print(set_userID)

set_recipeID = dict(zip(list(_part['recipe_id'].unique()),
                        list(range(len(_part['recipe_id'].unique())))))

clean_df = _part.replace({'user_id': set_userID, 'recipe_id': set_recipeID})

#print('The recipes without names: ', R['id'][R['name'].isnull()].values[0])
#print(clean_df[clean_df['recipe_id'] == R['id'][R['name'].isnull()].values[0]])

#to finish the remapping I select the columns pertinent to the dataset: name, id, and ingredients -- I merge or 'join' these columns to the newly created recipe_id 
recipe = R[['name', 'id', 'ingredients', 'steps']].merge(_part[['recipe_id']], 
                                                left_on = 'id', right_on = 'recipe_id', 
                                                how = 'right').drop(['id'], axis = 1).drop_duplicates().reset_index(drop = True)


# Ratings Assignment #

#centred cosine applied to ratings to make ratings more accurate
mean = clean_df.groupby(['user_id'], as_index = False, sort = False).mean().rename(columns = {'rating':'rating_mean'})
clean_df = clean_df.merge(mean[['user_id','rating_mean']], how = 'left')
clean_df.insert(2, 'rating_adjusted', clean_df['rating'] - clean_df['rating_mean'])

# DATASET TRAINING #
train_data, test_data = train_test_split(clean_df, test_size = 0.25)

n_users = clean_df.user_id.unique()
n_items = clean_df.recipe_id.unique()

#datasets split into 3:4 and 1:4 train and test data respectively
train_data_matrix = np.zeros((n_users.shape[0], n_items.shape[0]))
for row in train_data.itertuples():
    train_data_matrix[row[1]-1, row[2]-1] = row[3]

test_data_matrix = np.zeros((n_users.shape[0], n_items.shape[0]))
for row in test_data.itertuples():
    test_data_matrix[row[1]-1, row[2]-1] = row[3]

# COSINE SIMILARITY #
    
user_similarity = 1 - pairwise_distances(train_data_matrix, metric = 'cosine')

item_similarity = 1 - pairwise_distances(train_data_matrix.T, metric = 'cosine')

# PREDICTION #

def predict(ratings, similarity, _type = 'user'):
    if _type == 'user':
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis = np.newaxis)])
    
    elif _type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis = 1)]) 
    
    return pred

#user prediction 
user_pred = predict(train_data_matrix, user_similarity, _type = 'user')

user_pred_df = pd.DataFrame(user_pred, columns = list(n_items))
user_pred_df.insert(0, 'user_id', list(n_users))

#item prediction
item_pred = predict(train_data_matrix, item_similarity, _type = 'item')
item_pred_df = pd.DataFrame(item_pred, columns = list(n_items))
item_pred_df.insert(0, 'user_id', list(n_users))

# EVALUATION OF PREDICTIONS #

#Evaluation done by root mean square error

def RMSE(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    
    return sqrt(mean_squared_error(prediction, ground_truth))
user_RMSE = RMSE(user_pred, test_data_matrix)
item_RMSE = RMSE(item_pred, test_data_matrix)
print('user_RMSE = {}'.format(user_RMSE))
print('item_RMSE = {}'.format(item_RMSE))

#t-test
#willcoxin
#nonparametric equivalent

# ENGINE #

def getRecommendations_UserBased(user_id, top_n = 10):
    recipe_reccs = {}
    for old_user, new_user in set_userID.items():
        if user_id == new_user:
            print(f'Top {top_n} Recommended Recipes for Original User ID: {old_user}\n')
    
    recipe_rated = list(clean_df['recipe_id'].loc[clean_df['user_id'] == user_id])
    _all = user_pred_df.loc[user_pred_df['user_id'] == user_id].copy()
    _all.drop(user_pred_df[recipe_rated], axis = 1, inplace = True)
    unwatch_sorted = _all.iloc[:,1:].sort_values(by = _all.index[0], axis = 1, ascending = False)
    dict_top_n = unwatch_sorted.iloc[:, :top_n].to_dict(orient = 'records')

    i = 1
    for recipe_id in list(dict_top_n[0].keys()):
        for old_recipe, new_recipe in set_recipeID.items():
            if recipe_id == new_recipe:
                name = recipe[recipe['recipe_id'] == old_recipe]['name'].values[0]
                ingredients = recipe[recipe['recipe_id'] == old_recipe]['ingredients'].values[0]
                steps = recipe[recipe['recipe_id'] == old_recipe]['steps'].values[0]

                recipeToAdd = {
                    f'{i}': {
                        "ID" : new_recipe,
                        "name" : name,
                        "ingredients": ingredients,
                        "steps": steps
                    }
                }

                recipe_reccs.update(recipeToAdd)
                # recipe_rec = (f'Top {i} Original Recipe ID: {new_recipe} - {name}\n Ingredients: {ingredients}\n Steps: {steps}\n')
                
                i += 1
    print (recipe_reccs)
    return recipe_reccs

getRecommendations_UserBased(9960)


#ENUCC