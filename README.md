# Intro 

This Website was build to demonstrate our machine learning model. The front end
is a react webpage, which is communicating over a GET request to our python
backend. The backend is run on a server by flask and includes our trained 
machine learning model. 

# How to use it 
To run the backend enter in the terminal: npm run startFlask
To run the frontend enter in the terminal: npm run start
To run both at the same time enter: npm run startNEW

# Frontend Explanations

React is a open-source framwork, which allows to write a Webpage in 
Javascript and css.

The Front end contains multiple pages. The main Page index.js is rendered. Within
Index.js, we are calling App.js, which is the route webpage, which functions as 
a switch between our different pages. Our App has three working pages and one
Navigationbar: 

1. Home.js (Main page)
2. AboutUs.js (containing the presentation of our Business Idea as Screenshots)
3. Prediction (Page, which takes two parameters (longitude and latitude) and
calls the Machine Learning Model in the Python Backend. While calling the 
Backend we print the Steps on the Screen to visulise what is happening
in the Backend. When the API returns the prediction, it will also be shown on 
the Website. 
4. Navbar.js is the file which displayes the Navigationbar and contains the Link
to the different Pages

The index.css files contains all the code to align the different components 
on the website and make it visually nice.

The Package.json file is one of the most important files, as it is containing
all the requirements to run this website. In it we also defined the scripts 
how to let the Frond and Backend run.

# Backend Explanations

Flask is a tool which allows you to run a Python script on a server. This is 
needed, so that our Frontend can access the Python Machine Learning Model. All
files are included in the python-model file. In the file there are four main 
files worth mentioning: 

1. controller.py - this is the Backend connection Point to the Frontend. We have
a GET API Mathod, which waits for a call of the Frontend. If it receives a Call
with the given Langitude and Longitude, it activates the Demo() function, which 
calls the Database, retreives the information and runs the prediction. The Demo()
function is calling some subfunctions, also included in the model

2. solar.pickle - is our trained model

3. 2 Datasets - one dataset used to train the model, a second one for the demo
