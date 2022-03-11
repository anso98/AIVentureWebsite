import React from 'react';
import slide from './../Images/slide.png';
import backup from './../Images/BackupSlide.png'


function AboutUs(){

    return(
        <div className="aboutus">
            <img src={slide} alt="Slide1"/>;
            <br></br>
            <br></br>
            <img src={backup} alt="Slide2" />

        </div>
    );
}

export default AboutUs;