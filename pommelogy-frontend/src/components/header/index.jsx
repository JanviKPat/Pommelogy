import React from 'react'
import { NavLink, Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../contexts/authContext'
import { doSignOut } from '../../firebase/auth'
import './Header.css'
import profile from '../../assets/profile.png'
import search from '../../assets/search.png'

const Header = () => {
  const navigate = useNavigate()
  const { userLoggedIn } = useAuth()

  const handleLogout = async () => {
    await doSignOut()
    navigate('/login')
  }

  return (
    <header className='header'>
      <nav>
        <ul>
          <li>
            <NavLink to='/' exact activeClassName='active'>
              Home
            </NavLink>
          </li>
          <li>
            <NavLink to='/about' activeClassName='active'>
              About
            </NavLink>
          </li>
          <li>
            <NavLink to='/variety' activeClassName='active'>
              Variety
            </NavLink>
          </li>
          <li>
            <NavLink to='/contact' activeClassName='active'>
              Contact
            </NavLink>
          </li>
        </ul>
      </nav>
      <div className='header-icons'>
        <a href='/search'>
          <img src={search} alt='search' className='icon' />
        </a>
        {userLoggedIn ? (
          <button onClick={handleLogout} className='profile-button'>
            <img src={profile} alt='profile' className='icon' />
          </button>
        ) : (
          <Link to='/profile'>
            <img src={profile} alt='profile' className='icon' />
          </Link>
        )}
      </div>
    </header>
  )
}

export default Header
