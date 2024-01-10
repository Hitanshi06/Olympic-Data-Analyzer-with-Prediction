# import streamlit as st
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split



# st.title('Medal Prediction')

# st.header('Predicting the chances of winning a medal')

# st.code()
# # Import necessary libraries


# # Load the Olympics data
# df = pd.read_csv('athlete_events.csv')

# # Select relevant features
# X = df[['Age', 'Height', 'Weight', 'Season', 'Sex', 'Country']]
# y = df['Medal']

# # Convert categorical variables into numerical variables
# X = pd.get_dummies(X, columns=['Season', 'Sex', 'Country'], drop_first=True)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and fit the logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Predict the probability of winning a medal for each athlete in the test set
# y_pred_proba = model.predict_proba(X_test)[:, 1]

# # Set a threshold to predict if an athlete is likely to win a medal or not
# threshold = 0.5
# y_pred = (y_pred_proba >= threshold).astype(int)

# # Calculate the accuracy of the model
# accuracy = (y_pred == y_test).mean()
# print(f'Accuracy: {accuracy}')







# st.title('Medal Prediction')

# st.header('Predicting the chances of winning a medal')
# y = df['Medal']
# X = df.drop(['Medal'], axis=1)

# #Split data into training and testing data
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=1)

# Algo1 = 'LogisticRegression'
# lr = LogisticRegression(random_state=20, max_iter=1000)
# lr.fit(train_x, train_y)
# lr_predict = lr.predict(test_x)
# lr_conf_matrix = confusion_matrix(test_y, lr_predict)
# lr_acc_score = accuracy_score(test_y, lr_predict)

# # Define the user input form
# st.header("Prediction App")
# st.write("Enter the following parameters to make a prediction:")

# Algo1 = st.slider("Logistic Regression")
# # Algo2 = st.slider("Sepal width", 0.0, 10.0, 5.0, 0.1)
# # Algo3 = st.slider("Petal length", 0.0, 10.0, 5.0, 0.1)
# # Algo4 = st.slider("Petal width", 0.0, 10.0, 5.0, 0.1)

# # When the user clicks the "Predict" button, make a prediction and display the result
# if st.button("Predict"):
#     df = [Algo1]
#     prediction = lr.predict(test_x)
#     st.write(f"Prediction: {prediction}")



# st.title('Medal Prediction')

# st.header('Predicting the chances of winning a medal')

# Import necessary libraries


# Load the Olympics data
# df = pd.read_csv('athlete_events.csv')

# # To fill missing values in Medal column with 0's and 1's
# df['Medal'] = df['Medal'].apply(lambda x: 1 if str(x) != 'nan' else 0)

# # Checking null values in the data 
# df.isna().mean()

# # Fill null values with mean values for these columns
# for column in ['Age', 'Height', 'Weight']:
#     df[column] = df.groupby(['Medal', 'Sex'])[column].apply(lambda x: x.fillna(x.mean()).astype(np.int))


# # Checking null values again
# st.write("Total missing values:", df.isna().sum().sum())


# # st.write(df.info())

# # Select relevant features
# X = df[['Age', 'Height', 'Weight', 'Season', 'Sex']]
# y = df['Medal']

# # X = df.drop('Medal', axis=1)
# # y = df['Medal']

# # Convert categorical variables into numerical variables
# X = pd.get_dummies(X, columns=['Season', 'Sex'], drop_first=True)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and fit the logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Predict the probability of winning a medal for each athlete in the test set
# y_pred_proba = model.predict_proba(X_test)[:, 1]

# # Set a threshold to predict if an athlete is likely to win a medal or not
# threshold = 0.5
# y_pred = (y_pred_proba >= threshold).astype(int)

# # Calculate the accuracy of the model
# accuracy = (y_pred == y_test).mean()
# st.write(f'Accuracy: {accuracy}')

# algo2 = 'MultinomialNB'
# nv = MultinomialNB()
# nv.fit(X_train, y_train)
# nv_predict = nv.predict(X_train)
# nv_conf_matrix = confusion_matrix(y_test, nv_predict)
# nv_acc_score = accuracy_score(y_test, nv_predict)
# st.write("confusion matrix")
# st.write(nv_conf_matrix)
# st.write("\n")
# st.write("Accuracy of Logistic Regression:",nv_acc_score*100,'\n')
# st.write(classification_report(y_test,nv_predict))



#  if user_menu == 'Medal Prediction':
    #     # st.title('Medal Prediction')
#     years, country = helper.country_year_list(df)

# st.title('Predict The Winning Team')
# @st.cache(allow_output_mutation=True)
# def load_data(nrows):
#     data = pd.read_csv('athlete_events.csv', nrows=nrows)
#     return data

# df = load_data(271116)
# classifier_name = st.sidebar.selectbox(
#     'Select classifier',
#     ('Logistic Regression', 'Random Forest', 'Decision Tree')
# )

# selected_year = st.sidebar.selectbox("Select Year", years)
# selected_country = st.sidebar.selectbox("Select Country", country)



# # When the user clicks the "Predict" button, make a prediction and display the result
# if st.button("Predict"):
#     input_data = [df]
#     prediction = load_data(input_data)
#     st.write(f"Prediction: {prediction}")

# import streamlit as st
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

# # Check for missing values
# if df.isnull().sum().sum() > 0:
#     # Impute missing values with most frequent value
#     imputer = SimpleImputer(strategy='most_frequent')
#     df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#     df['Medal'] = df['Medal'].apply(lambda x: 1 if str(x) != 'nan' else 0)

# @st.cache(allow_output_mutation=True)
# def load_data(nrows):
#     data = pd.read_csv('athlete_events.csv', nrows=nrows)
#     return data

# # Set the filtering criteria
# # year = 2016
# # country = 'USA'
# years, country = helper.country_year_list(df)
# # Filter the data
# filtered_df = df[(df['Year'] == years) & (df['NOC'] == country)]

# # Check if the filtered DataFrame is empty
# if filtered_df.empty:
#     print('No data found for the selected year and country')
# else:
#     # Split the data into features and target
#     X = filtered_df[['Age', 'Height', 'Weight']]
#     y = filtered_df['Medal']

#     # Replace missing values with the mean
#     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#     X_imp = imp.fit_transform(X)

#     # Train a logistic regression model
#     model = LogisticRegression()
#     model.fit(X_imp, y)


# def medal_prediction(df, year, country, classifier):
#     # Filter the data by year and country
#     data = df[(df['Year'] == year) & (df['NOC'] == country)]
    
#     # Split the data into features and target
#     X = data[['Age', 'Height', 'Weight']]
#     y = data['Medal']
    
#     # Train a classifier on the data
#     if classifier == 'Logistic Regression':
#         model = LogisticRegression()
#     elif classifier == 'Random Forest':
#         model = RandomForestClassifier()
#     elif classifier == 'Decision Tree':
#         model = DecisionTreeClassifier()
    
#     model.fit(X, y)
    
#     # Make a prediction and return the result
#     prediction = model.predict([[25, 170, 70]])[0]
#     return prediction

# # Load the data
# df = load_data(271116,12)

# # Create the app
# st.title('Predict The Winning Team')
# classifier_name = st.sidebar.selectbox(
#     'Select classifier',
#     ('Logistic Regression', 'Random Forest', 'Decision Tree')
# )
# years = sorted(df['Year'].unique())
# selected_year = st.sidebar.selectbox("Select Year", years)
# countries = sorted(df['NOC'].unique())
# selected_country = st.sidebar.selectbox("Select Country", countries)

# if st.button("Predict"):
#     prediction = medal_prediction(df, selected_year, selected_country, classifier_name)
#     st.write(f"Prediction: {prediction}")
# 

# if classifier_name == "Logistic Regression":
#     X = df.drop(['Medal'], axis=1)
#     y = df['Medal']
#     model = LogisticRegression()
#     model.fit(X, y)
# def medal_prediction(selected_country, model):
    
    
#     return medal_prediction


#  if st.button("Prediction"):
    #     X = df[['Year']]
#     y = df['Medal']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     model = LogisticRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     st.write('Accuracy:', accuracy)



# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor

# # Load the Olympics dataset
# # data_url = "https://raw.githubusercontent.com/rahuljaiswaniitg/PythonProjects/main/olympic_dataset.csv"
# # df = pd.read_csv(data_url)

# # Set up the Streamlit app
# st.title("Olympics Medal Prediction")
# st.markdown("### Predicting medal count ")

# # Create a function to split the dataset and train a model
# def train_model(df, feature_cols, target_col, model_type):
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)

#     # Train the model
#     if model_type == "Linear Regression":
#         model = LinearRegression()
#     elif model_type == "Random Forest":
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#     else:
#         st.error("Invalid model type selected!")
#         return None

#     model.fit(X_train, y_train)
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)

#     st.write("Training Score:", train_score)
#     st.write("Testing Score:", test_score)

#     return model

# # Create a list of available features and targets
# features = ["Year", "City", "Sport"]
# targets = [ "Gold", "Silver", "Bronze"]

# # Create a dropdown to select the feature and target columns
# feature_col = st.sidebar.selectbox("Select a feature column", features)
# target_col = st.sidebar.selectbox("Select a target column", targets)

# # Create a radio button to select the model type
# model_type = st.sidebar.radio("Select a model type", ["Linear Regression", "Random Forest"])

# # Train the model and display the results
# model = train_model(df, [feature_col], target_col, model_type)
# if model is not None:
#     year_list = list(range(2020, 2025))
#     years = st.selectbox('Select a year', year_list)
#     # min_year = int(df["Year"].min())
#     # max_year = int(df["Year"].max())
#     # year = st.slider("Select a year", min_value=min_year, max_value=max_year)

#     # year = st.slider("Select a year", min_value=df["2017"].min(), max_value=df["2023"].max())
#     # selected_country = st.selectbox("Select Country", country)
#     host_country = st.selectbox("Select Country", country)
#     # summer_winter = st.radio("Summer or Winter Olympics?", ["Summer", "Winter"])
#     # sport = st.text_input("Enter the sport")
#     sport_list = df['Sport'].unique().tolist()
#     sport_list.sort()
#     # sport_list.insert(0, 'Overall')
#     sport = st.selectbox('Select a Sport', sport_list)

#     input_data = pd.DataFrame({
#         "Year": [years],
#         "City": [host_country],
#         # "Summer/Winter": [summer_winter],
#         "Sport": [sport_list]
#     })
    
#     prediction = model.predict(input_data[[feature_col]])
#     st.button("Prediction")
#     st.write("Predicted medal count for", target_col, "in", years, "for", sport, ":", int(prediction))


# Prediction = medal_prediction(selected_country,selected_year,selected_sport, model)
# st.write(f"Predicted medal count for {selected_country}: {Prediction}")


# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor

# # Load the Olympics dataset
# # data_url = "https://raw.githubusercontent.com/rahuljaiswaniitg/PythonProjects/main/olympic_dataset.csv"
# # df = pd.read_csv(data_url)

# # Set up the Streamlit app
# st.title("Olympics Medal Prediction")
# st.markdown("### Predicting medal count using Random Forest and Linear Regression")
# df.dropna(inplace=True)
# # Create a function to split the dataset and train a model
# def train_model(df, feature_cols, target_col, model_type):
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)

#     # Train the model
#     if model_type == "Linear Regression":
#         model = LinearRegression()
#     elif model_type == "Random Forest":
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#     else:
#         st.error("Invalid model type selected!")
#         return None

#     model.fit(X_train, y_train)
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     # y_pred = model.predict(X_test)
#     # accuracy = r2_score(y_test, y_pred)
#     y_pred = model.score(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     st.write("Training Score:", train_score)
#     st.write("Testing Score:", test_score)
#     # st.write('Accuracy:', accuracy)

#     return model

# # Create a list of available features and targets
# features = ["Year", "City", "Sport"]
# targets = [ "Gold", "Silver", "Bronze"]

# # Create a dropdown to select the feature and target columns
# feature_col = st.sidebar.selectbox("Select a feature column", features)
# target_col = st.sidebar.selectbox("Select a target column", targets)

# # Create a radio button to select the model type
# model_type = st.sidebar.radio("Select a model type", ["Linear Regression", "Random Forest"])

# # Train the model and display the results
# model = train_model(df, [feature_col], target_col, model_type)
# if model is not None:
#     year_list = list(range(2020, 2025))
#     year = st.selectbox('Select a year', year_list)
#     # year = st.selectbox("Select a year", min_value=df["Year"].min(), max_value=df["Year"].max())
#     host_country = st.selectbox("Select the country",country)
#     # summer_winter = st.radio("Summer or Winter Olympics?", ["Summer", "Winter"])
#     sport = st.text_input("Enter the sport")

#     input_data = pd.DataFrame({
#         "Year": [year],
#         "City": [host_country],
#         # "Summer/Winter": [summer_winter],
#         "Sport": [sport]
#     })

#     prediction = model.predict(input_data[[feature_col]])
#     st.write("Predicted medal count for", target_col, "in", year, "for", sport, ":", int(prediction))

# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor

# # Load the Olympics dataset
# # data_url = "https://raw.githubusercontent.com/rahuljaiswaniitg/PythonProjects/main/olympic_dataset.csv"
# # df = pd.read_csv(data_url)

# # Set up the Streamlit app
# st.title("Olympics Medal Prediction")
# st.markdown("### Predicting medal count using Random Forest and Linear Regression")
# df.dropna(inplace=True)
# # Create a function to split the dataset and train a model
# def train_model(df, feature_cols, target_col, model_type):
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)

#     # Train the model
#     if model_type == "Linear Regression":
#         model = LinearRegression()
#     elif model_type == "Random Forest":
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#     else:
#         st.error("Invalid model type selected!")
#         return None

#     model.fit(X_train, y_train)
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     # y_pred = model.predict(X_test)
#     # accuracy = r2_score(y_test, y_pred)
#     y_pred = model.score(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     st.write("Training Score:", train_score)
#     st.write("Testing Score:", test_score)
#     # st.write('Accuracy:', accuracy)

#     return model

# # Create a list of available features and targets
# features = ["Year", "City", "Sport"]
# targets = [ "Gold", "Silver", "Bronze"]

# # Create a dropdown to select the feature and target columns
# feature_col = st.sidebar.selectbox("Select a feature column", features)
# target_col = st.sidebar.selectbox("Select a target column", targets)

# # Create a radio button to select the model type
# model_type = st.sidebar.radio("Select a model type", ["Linear Regression", "Random Forest"])

# # Train the model and display the results
# model = train_model(df, [feature_col], target_col, model_type)
# if model is not None:
#     year_list = list(range(2020, 2025))
#     year = st.selectbox('Select a year', year_list)
#     # year = st.selectbox("Select a year", min_value=df["Year"].min(), max_value=df["Year"].max())
#     host_country = st.selectbox("Select the country",country)
#     # summer_winter = st.radio("Summer or Winter Olympics?", ["Summer", "Winter"])
#     sport = st.text_input("Enter the sport")

#     input_data = pd.DataFrame({
#         "Year": [year],
#         "City": [host_country],
#         # "Summer/Winter": [summer_winter],
#         "Sport": [sport]
#     })

#     prediction = model.predict(input_data[[feature_col]])
#     st.write("Predicted medal count for", target_col, "in", year, "for", sport, ":", int(prediction))







# country = helper.c_list(df)
# year = helper.year_list(df)
# sport = helper.sport_list(df)
# modelf= None

# if user_menu == 'Medal Prediction':
#     st.sidebar.title("Medal Prediction") 
#     year_list = list(range(2017, 2023))
#     year = st.selectbox('Select a year', year_list, key = "year")
#     country = st.selectbox("Select Country", country, key = "country")
#     sport = st.selectbox('Select a Sport', sport, key = "sport")
#     age = st.text_input("Enter the age")
#     height = st.text_input("Enter the height")
#     weight = st.text_input("Enter the weight")

#     classifier_name = st.selectbox(
#         'Select classifier',
#         ('Overall', 'Logistic Regression', 'Random Forest', 'Decision Tree'),
#         key = "classifier"
#     )


#     # Display the prediction results in the Streamlit app

#     # st.button("Prediction")
#     st.button("Prediction", key="medal_prediction")
#     # X = df[['Year']]
#     X = df[['Age', 'Height', 'Weight','Sex', 'City','Sport']]
#     y = df['Medal']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     modelf = LogisticRegression()
#     modelf.fit(X_train, y_train)
#     # model.save("./model/mymodel.h5")
#     pickle.dump(modelf, open('model.pkl', 'wb'))
#     y_pred = modelf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     st.write('Accuracy:', accuracy)

# def medal_prediction(country, years, sport, model):
#     # Filter the dataframe to include only the specified country, year, and sport
#     # pandas.DataFrame(data, index, columns)
#     filtered_df = df[(df['Team'] == country) & (df['Year'] == years) & (df['Sport'] == sport)]
#     print("dataframe here",filtered_df)

#     if filtered_df.empty:
#         st.write("No data found for the given input.")
#     else:
#         # Get the input features for the selected rows
#         X = pd.DataFrame([year],columns=["Year"])
#         print("X here",X)

#         if st.button("Prediction"):
#             # Make predictions using the trained model
#             predictions = model.predict(X)
#             st
#             # Count the number of each medal type in the predictions
#             medal_counts = {'Gold': 0, 'Silver': 0, 'Bronze': 0}
#             for prediction in predictions:
#                 medal_counts[prediction] += 1

#             # Print the predicted medal counts for the user
#             st.write(f"Predicted medal count for {country} in {years} ({sport}):")
#             st.write(f"Gold: {medal_counts['Gold']}")
#             st.write(f"Silver: {medal_counts['Silver']}")
#             st.write(f"Bronze: {medal_counts['Bronze']}")   

# #--------------------------------------------------------------------------------------------------
# Prediction = medal_prediction(country, year, sport, modelf)
# st.write(f"Predicted medal count for {country} in {year} for {sport}: {Prediction}")

# # st.write("Classification Report:\n\n", classification_report(y_test, y_pred))

# # Prediction = medal_prediction(selected_country,selected_year,selected_sport, model)
# # st.write(f"Predicted medal count for {selected_country}: {Prediction}")
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Load the data
# olympics_data = pd.read_csv("athlete_events.csv")
# olympics_data.fillna(olympics_data.mean(), inplace=True)
# # st.write(df.head())
# # st.write(olympics_data.isnull().sum())
# # olympics_data.to_csv("D:\SEM I-VII\SEM-VII\New folder\Olympic--Data_Analysis-Web-App-main\Olympic--Data_Analysis-Web-App-main\athlete_events.csv", index=False)
# # df = df.drop(['ID', 'Name', 'Games', 'region', 'notes'], axis=1)
# # olympics_data.drop(["region","notes"])
# # st.write(df.isnull().sum())
# # Create a function to train the model and predict the medal
# def predict_medal(country, year, sport):
#     # Filter the data based on the given inputs
#     data = olympics_data[(olympics_data['Year'] == year) & (olympics_data['Team'] == country) & (olympics_data['Sport'] == sport)]
#     st.write("Shape of the data: ", data.shape)
#     X = data[['Age', 'Height', 'Weight']]
#     y = data['Medal']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     # Train the logistic regression model
#     lr = LogisticRegression()
#     lr.fit(X_train, y_train)
#     pickle.dump(lr, open('model.pkl', 'wb'))
#     # Predict the medal
#     prediction = lr.predict(X_test)
#     # Calculate the accuracy of the prediction
#     accuracy = accuracy_score(y_test, prediction)
#     return accuracy

# # Set up the Streamlit app
# st.title("Medal Prediction for Olympics")
# st.markdown("Enter the details below to predict the medal:")

# # Create the input fields
# country = st.text_input("Country")
# year = st.slider("Year", 2020, 2024)
# sport = st.selectbox("Sport", olympics_data['Sport'].unique())

# # When the user clicks the 'Predict' button
# if st.button("Predict"):
#     # Call the predict_medal function and display the result
#     accuracy = predict_medal(country, year, sport)
#     st.write("The predicted accuracy of winning a medal for {} in {} for {} is {:.2f}%".format(country, year, sport, accuracy*100))
