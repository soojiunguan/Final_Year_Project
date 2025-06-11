# Abstract
This project presents the development of a personalized dietary recommendation system using unsupervised machine learning techniques. By analyzing a large-scale recipe dataset and allergen information, the system clusters recipes based on their nutritional profiles to deliver tailored meal suggestions that align with individual health goals and dietary restrictions. Extensive data preprocessing, feature engineering, and anomaly detection were performed to ensure data quality. Several clustering algorithms, including K-Means, MiniBatch K-Means, Gaussian Mixture Models, and Self-Organizing Maps, were implemented and evaluated using internal validation metrics such as Davies-Bouldin Index, Calinski-Harabasz Score, Bayesian Information Criterion, and Quantization Error. K-Means with MinMaxScaler preprocessing and seven clusters was selected as the optimal model. The final model was deployed through a Streamlit web application, featuring user profiling, calorie and macronutrient calculations, personalized recipe recommendations, downloadable meal plans, and integration with Google Gemini AI for additional health advice. Through this multi-phase approach, the project successfully delivers a flexible, inclusive, and health-centered dietary recommendation system aligned with Sustainable Development Goal 3.

Dataset: 

RecipeData.csv: https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews/data

AllergenData.csv: https://www.kaggle.com/datasets/boltcutters/food-allergens-and-allergies
