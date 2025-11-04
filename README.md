# OIBSIP_domain_task4
Task 4 : EMAIL SPAM DETECTION WITH MACHINE LEARNING
📧 Project Title: Spam Message Detection using Machine Learning.
🎯 Objective
   To develop a machine learning model that classifies text messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques.
🧠 Steps Performed
   Import Libraries – Loaded essential libraries for data analysis, visualization, and model building.
   Load Dataset – Imported the spam.csv dataset containing text messages labeled as spam or ham.
   Data Cleaning – Selected relevant columns (v1, v2) and renamed them to label and message.
   Label Encoding – Converted text labels (ham, spam) into numerical values (0, 1).
   Data Splitting – Divided the dataset into training and testing sets (80%-20%).
   Text Vectorization – Used TF-IDF Vectorizer to convert text data into numerical form.
   Model Training – Trained a Multinomial Naive Bayes model on the training data.
   Prediction & Evaluation – Predicted outcomes for the test set and evaluated performance using Accuracy, Classification Report, and Confusion Matrix.
   Visualization – Displayed a heatmap of the confusion matrix for better interpretation.
   Custom Prediction – Tested the model on a sample message to check spam detection.
⚙️ Tools & Libraries Used
   Python
   Pandas – Data handling
   NumPy – Numerical operations
   Matplotlib & Seaborn – Data visualization
   Scikit-learn – Machine learning algorithms & metrics
📊 Output:
   The model achieved high accuracy in distinguishing spam from ham messages.
   Confusion Matrix visualizes correct and incorrect predictions.
Example:
   Input: “Congratulations! You have won $1000. Click here to claim your prize.”
   Output: Spam
