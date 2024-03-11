from webapp._init_ import create_app
from flask import Blueprint, render_template, request, redirect, url_for, flash
from testing import getRecommendations_UserBased

application = create_app()

recipes = getRecommendations_UserBased(9960) 


#recipes provides ingredients and steps as a string, this for loop converts them into lists
for recipe in recipes:
        curr_recipe = recipes.get(recipe)
        ingredients = curr_recipe.get('ingredients')
        ingredients = ingredients.translate({ord("["): None})
        ingredients = ingredients.translate({ord("]"): None})
        ingredients = ingredients.split("'")
        for x in ingredients:
             if len(x) < 3:
                  ingredients.remove(x)
        curr_recipe['ingredients'] = ingredients
        steps = curr_recipe.get('steps')
        steps = steps.translate({ord("["): None})
        steps = steps.translate({ord("]"): None})
        steps = steps.split("'")
        for x in steps: 
             if len(x) < 3:
                  steps.remove(x)
        curr_recipe['steps'] = steps

    

@application.route('/')
def index():
    
    return render_template('index.html', recipes = recipes)

if __name__ == '__main__':
    application.run(debug=True)