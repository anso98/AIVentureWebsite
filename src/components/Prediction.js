import { useState } from "react";
import React from 'react';
import '../App.css';
import resultGraph from "./../Images/graph.jpeg";


/* Webpage where we imput Longitude and Latitude and enable the Website to call the python backend via flask. After the backend runs the machine learning model, it return a value which will be shown on the page */
const Prediction = () => {

  /* Defining variables to iterativly load the page */  
  const [prediction, setPrediction] = useState([]);
  const [isPending, setIsPending] = useState(0);
  const [isPrinting, setIsPrinting] = useState(false);

  /* Defining variables to store the Output */
  const [predResult, setPredResult] = useState();

    /* Function which is called when user hits predict button. Calls run Model Function which calls the Machine Learning */
    const handlePrediction = async () => {
    
    setIsPending(1);

    /* Function which calls Python Model API */
    runMLModel();
    
    /* Returning Print Statements */
    await sleep(1000);
    const newPrediction = {title: "1. Collecting weather data for location", id: 1};
    setPrediction(prediction => [newPrediction]);
    
    setIsPrinting(true);

    await sleep(1000);
    const newPrediction1 = {title: "2. Pre-processing weather data", id: 2};
    setPrediction(prediction => [prediction[0], newPrediction1]);

    await sleep(1000);
    const newPrediction2 = {title: "3. Using pre-trained neural network to analyze data", id: 3};
    setPrediction(prediction => [prediction[0], prediction[1], newPrediction2]);

    await sleep(1000);
    const newPrediction3 = {title: "4. Making final prediction for solar energy potential", id: 4};
    setPrediction(prediction => [prediction[0], prediction[1], prediction[2], newPrediction3]);
    await sleep(2000);
    setIsPending(2);
    await sleep(2000);
    setIsPending(3);
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
    }

    /* Function which calls the Python Backend and return the value after the API returned the respective value */
    async function runMLModel (){
        var long = document.getElementById("long").value;
        var lat = document.getElementById("lat").value;
        var coordinates = {
            'long': long,
            'lat': lat,
        }

        /* Request to the Backend  */
        const request = new XMLHttpRequest()
        request.open('POST', '/pythonScript/' + JSON.stringify(coordinates), true)
        request.setRequestHeader('Content-Type', 'application/json; charset=UTF-8')
        console.log("Right before sending");
        request.send()

        /* Process answer from backend */
        request.onreadystatechange = () => {
            console.log("Check if Statement");
            if (request.readyState == 4 && request.status == 200) {
                var result = request.response;
                console.log(result)
                setPredResult(result);

            }
        }
    }

/* Website code */
  return (
    <div className="Prediction">
        <div className="InputFields">
            <h2> Please insert your location: </h2> 
            <br></br>
            <label>Latitude: </label>
            <input type="text" name="Latitude" placeholder="Latitude" id="lat" required></input>
            <br></br>
            <label>Longitude: </label>
            <input type="text" name="Longitude" placeholder="Longitude" id="long" required></input>
            <br></br>
            <button onClick={() => handlePrediction()}>predict</button>
            <br></br>
        </div>
        
        <div className="ProgressReport">
            {isPending === 1 && <div className="Stage1"> Prediction in Progress.... </div>}
            {(isPending === 2 || isPending === 3) && <div className="Stage2"> Prediction completed! </div>}
            <br></br>
            {isPrinting && <div className="progressStatementBox">
                {prediction.map(prediction => (
                <div className="oneStepPreview" key={prediction.id} >
                    { prediction.title }
                </div>
                ))}
            </div>}
        </div>

        {(isPending === 2 || isPending === 3) && <div className="ResultDisplay">
             <div className="TextResult">
                <h2> Your Estimate:</h2>
                A solar panel at your location will produce
                <div className="highlight"> {predResult} kWh </div>
                every year!
                ( R² of 86%)
            </div>
            {isPending === 3 && <img src={resultGraph} alt="Graph"/>}
        </div>}
    </div>
  );
}
 
export default Prediction;
