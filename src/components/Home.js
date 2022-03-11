import React from 'react';
import Background from './../Images/background.png';


function Home(){

    return(
        <div className="home">
            <h1> Welcome to Sunshine Avenue </h1>
            <h3> The revolutionary Solarpanel prediction company</h3>
            <br></br>
            <br></br> 
            <div style={{ 
                backgroundImage: 'url('+ {Background} + ')',
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'center',
                backgroundSize: 'cover',
            }}></div>
            <img src={Background} alt="Background"/>
            <br></br>
        </div>
    );
}

export default Home;