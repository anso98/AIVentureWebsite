import './App.css';
import Navbar from './components/Navbar';
import Home from './components/Home';
import Prediction from './components/Prediction';
import AboutUs from './components/AboutUs';
import React from 'react';
import { BrowserRouter as Router , Route, Routes} from 'react-router-dom';


function App() {

  // WORKING API CALL TO OPENWEATHER - USE TO BUILD DATASET - Currently not actived
  async function apiCall() {
    let lat = 53.483959;
    let lon = -2.244644;
    var start = 1617235200;
    var end = 1638403200;
    var myappid = "725dbe7a9adf94afad40d9d686cd5988";


    var URI = 'http://history.openweathermap.org/data/2.5/history/city?lat=' + lat + '&lon=' +lon + '&type=hour&start=' +start + '&end=' +end + '&appid=' + myappid;
    let response = await fetch(URI)
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(err => console.error(err));
  }

  //apiCall();


  return (
    <Router>
      <div className="App">
      <Navbar></Navbar>
      <div className="content"> 
      <Routes>
        <Route exact path="/" element={<Home />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/about-us" element={<AboutUs />} />
      </Routes>
      </div>
    </div>
    </Router>
  );
}

export default App;
