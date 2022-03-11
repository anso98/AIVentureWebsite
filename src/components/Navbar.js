import React from 'react';
import {Link} from 'react-router-dom';

function Navbar(){

    /* Website Code */
    return(
        <nav className='navbar'>
             <Link to="/" > <h1> Sunshine Avenue </h1> </Link>
            <div className='links'>
                <Link to="/prediction"> Prediction </Link>
                <Link to="/about-us" > About us </Link>
            </div>
        </nav>
    )
}

export default Navbar;