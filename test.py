import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np  
data=pd.read_csv('IndianFoodDatasetCSV.csv')

data['Course'].unique()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
data.dropna(inplace=True)
def preprocess_keywords(keywords):
    keywords = keywords.lower()
    keywords = keywords.replace(",", "")
    return keywords  # Return a string, not a list


def recipe_recommender(keywords):
    preprocessed_keywords = preprocess_keywords(keywords)
    all_text = data['TranslatedIngredients'].tolist()
    lowered_text = [text.lower() for text in all_text]
    lowered_text.append(preprocessed_keywords)  # Append the preprocessed keywords string
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(lowered_text)
    user_vector = matrix[-1]
    cosine_similarities = linear_kernel(user_vector, matrix[:-1])  # Compare with all recipes except the last one (user input)
    food_ind = similar_recipes(cosine_similarities[0])
    return data['TranslatedRecipeName'].iloc[food_ind]

def similar_recipes(similarities):
    sim_score = list(enumerate(similarities))
    sim_score = sorted(sim_score, key=lambda x:x[1], reverse=True)
    sim_score = sim_score[:30]  # Get top 30 similar recipes
    food_ind = [i[0] for i in sim_score]
    return food_ind

keywords = input("Enter food item")
recommendations = recipe_recommender(keywords)

print("Recommended recipes:")
print(recommendations)

from sklearn.preprocessing import LabelEncoder
Lableencoder = LabelEncoder()
le=LabelEncoder()
data['Course_encoded'] = Lableencoder.fit_transform(data['Course'])
data['Cuisine_encoded']= le.fit_transform(data['Cuisine'])

mapping = dict(zip(Lableencoder.classes_, range(len(Lableencoder.classes_))))
print(mapping)

mapping1 = dict(zip(le.classes_, range(len(le.classes_))))
print(mapping1)

from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)

data['is_train']=np.random.uniform(0,1,len(data))<=.75
data.head()

train , test = data[data['is_train']==True], data[data['is_train']==False]
print('Train', len(train))
print('Test ', len(test))

features = ['CookTimeInMins', 'TotalTimeInMins', 'Servings','Course_encoded','Cuisine_encoded']
selected_data = data[features]

selected_data

y= pd.factorize(train['Course_encoded'])[0]
y

# Check if features has only one element
if len(features) == 1:
    selected_data = train[features[0]]
else:
    selected_data = train[features]  # No need for to_frame() here



clf=RandomForestClassifier(n_jobs=2, random_state=0)
y, _ = pd.factorize(train['Course_encoded'])  # Extract only the encoded labels (y)
clf.fit(selected_data, y)

vAL = clf.predict(test[features])

print(vAL[:100])

# Assuming you have a list of class names for your dataset
class_names = ['Side Dish', 'Main Course', 'South Indian Breakfast', 'Lunch','Snack', 'High Protein Vegetarian', 'Dinner', 'Appetizer',
                'Indian Breakfast', 'Dessert', 'North Indian Breakfast',
                'One Pot Dish', 'World Breakfast', 'Non Vegeterian', 'Vegetarian',
                'Eggetarian', 'No Onion No Garlic (Sattvic)', 'Brunch', 'Vegan',
                'Sugar Free Diet']

predicted_indices = clf.predict(test[features])
predicted_labels = [class_names[i] for i in predicted_indices]

