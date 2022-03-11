# Intro
This website was created to demonstrate our machine learning model. The frontend
is a React web page that communicates via a GET request with our Python backend. The backend is running on a server with Flask and contains our trained machine learning model. 

# How to use it 
To install the environment:
You need to install a package manager (npm) and node on your computer
You can do so, by installing it over nvm (a node version managers) 
Please find all information here: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm
Afterwards, in the folder of the website, you can run npm install , which will install all dependencies needed for the react app (they are specified in package.json and will load automatically after running the command)

To start the App:
In order to run the full Website, you need to start both Backend and Frontend:
To start the backend, type into the terminal: npm run startFlask.
To start the frontend, type in the terminal: npm run start
To start both at the same time, type: npm run startNEW

# Frontend explanations
React is an open source framework that allows you to write a website in Javascript and css.

The frontend contains several pages. The main index.js page is rendered. Within Index.js we call App.js, which is the route web page that acts as a switch between our different pages. Our app has three main pages and a navigation bar: 

1. Home.js (main page)
2. AboutUs.js (contains the presentation of our business idea as screenshots)
3. Prediction.js (page that asks the user for two parameters (longitude and latitude) and calls the machine learning model in the Python backend. While calling the backend we print out the steps on the screen to visualize what is happening in the backend. When the API returns the prediction, it is displayed on the website.) 
4. navbar.js (is the file that displays the navigation bar and provides the link to the different pages)

The index.css file contains all the code to align the different components on the website and to make them visually appealing.

The package.json file is one of the most important files because it contains all the requirements for running the website. In this file we have also defined the scripts how to make the frontend and backend run.

# Backend Explanations
Flask is a tool that allows you to run a Python script on a server. This is needed for our frontend to access the Python Machine Learning Model. All files are contained in the python-model file. In the file there are two important files that are worth mentioning: 

1. controller.py (this is the connection point between the backend and the frontend. We created a GET API method that waits for a call from the frontend. When it receives a call
with the latitude and longitude specified, it activates the function Demo(), which calls the database, retrieves the information and performs the prediction. The Demo() function calls some sub-functions that are also included in the file.)

2. solar.pickle (is our trained model)

Additionally, the folder holds two datasets, one dataset was used to train the model (and to create the solar.pickle file) and the second model is the new data we use to predict energy output for a specific location.
