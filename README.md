# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: RAMA KANDIVALASA 

*INTERN ID*: CT04DN904

*DOMAIN*: MACHINE LEARNING 

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

## 📌 Project Title: Building a Decision Tree Model using Scikit-Learn on the Iris Dataset
🔍 Introduction

This project demonstrates the application of a Decision Tree Classification algorithm using the Scikit-learn library on the popular Iris dataset. The project was implemented in Jupyter Notebook, which is widely used for data science tasks due to its interactivity, easy visualization support, and seamless integration with Python libraries.

The goal of this task was to classify different species of iris flowers (Setosa, Versicolor, and Virginica) based on four flower features: sepal length, sepal width, petal length, and petal width. The project involves a complete pipeline including data loading, preprocessing, model training, evaluation, and visualization of the decision tree.

🔧 Tools and Libraries Used

Python – Core programming language

Jupyter Notebook – Development environment for running and visualizing code

Pandas – Used for data manipulation and handling tabular data

NumPy – Used for numerical operations and array handling

Matplotlib & Seaborn – Used for plotting and data visualization

Scikit-learn (sklearn) – Machine learning library used for:

Loading datasets

Model training and evaluation

Visualizing decision trees

📂 Dataset

The project utilizes the Iris dataset, which is a built-in dataset in Scikit-learn. It contains:

150 rows (samples)

4 features (columns):

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Target variable: The class of iris flower (0: Setosa, 1: Versicolor, 2: Virginica)

🔄 Project Workflow

📌 Step 1: Importing Required Libraries

The project begins by importing all necessary libraries such as pandas, numpy, matplotlib.pyplot, and machine learning tools from sklearn.

📌 Step 2: Loading the Dataset

Using load_iris() from sklearn.datasets, the Iris dataset is loaded into memory. A DataFrame is created with feature names as column headers and a new column called target is added for the flower class.

📌 Step 3: Splitting the Dataset

The dataset is split into features (X) and target (y). Then, using train_test_split(), the data is split into training (80%) and testing (20%) sets to evaluate the model's performance on unseen data. A random_state=42 ensures reproducibility of results.

📌 Step 4: Building the Decision Tree Model

The decision tree classifier is created using DecisionTreeClassifier() with the criterion set to 'gini' for Gini impurity. The model is trained using fit() on the training data.

📌 Step 5: Predictions and Evaluation

The model predicts the labels on the test set using predict(). Its performance is then evaluated using three key metrics:

Accuracy Score: Measures the overall correctness of the model

Classification Report: Shows precision, recall, f1-score for each class

Confusion Matrix: Shows the number of true/false positives and negatives, helping understand misclassifications

📌 Step 6: Visualization

Finally, plot_tree() from sklearn.tree is used to graphically visualize the trained decision tree. This step helps understand the decision rules, splits, and hierarchy used by the model to classify the flowers.

📈 Results and Insights

The model achieved a high accuracy, showing it effectively learned the distinctions between the three species.

The confusion matrix and classification report confirm minimal misclassifications, especially for Setosa which is linearly separable.

The decision tree visualization showed how petal length and width play a critical role in decision-making, with clear decision boundaries.

💡 Applications of Decision Trees

Medical Diagnosis: Classifying diseases based on symptoms or test values.

Finance: Credit risk evaluation or loan default prediction.

Retail: Customer segmentation or product recommendation.

Agriculture: Crop disease prediction or yield forecasting.

In this task, decision trees were used for a supervised classification problem, but they are also adaptable to regression tasks (e.g., DecisionTreeRegressor).

✅ Conclusion

This project successfully implemented a Decision Tree Classification Model using Scikit-learn in Jupyter Notebook. It involved the complete machine learning pipeline—data loading, preprocessing, model training, evaluation, and visualization. Decision trees are intuitive, interpretable, and powerful for small to medium-sized datasets like Iris. The visual tree representation not only helps in understanding model decisions but also in explaining the logic behind predictions to non-technical stakeholders.

This foundational exercise sets the stage for exploring advanced ensemble methods like Random Forests and Gradient Boosted Trees, which build on single decision trees for better performance and generalization.

## OUTPUT
Accuracy: 1.0

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           
           1       1.00      1.00      1.00         9
           
           2       1.00      1.00      1.00        11
           
    accuracy                           1.00        30
    
   macro avg       1.00      1.00      1.00        30
   
weighted avg       1.00      1.00      1.00        30


Confusion Matrix:

 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
 
![Image](https://github.com/user-attachments/assets/e568ce04-2587-4a10-963c-6fc7ded39a37)
