import React, { useState } from 'react'
import { Navigate, Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../../contexts/authContext'
import { doCreateUserWithEmailAndPassword } from '../../../firebase/auth'
import './SignUp.css'
import {
  FaUser,
  FaEnvelope,
  FaLock,
  FaEye,
  FaEyeSlash,
  FaExclamationCircle,
} from 'react-icons/fa'

const SignUp = () => {
  const navigate = useNavigate()

  const [showPassword1, setShowPassword1] = useState(false)
  const [showPassword2, setShowPassword2] = useState(false)
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [isRegistering, setIsRegistering] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
   const [firstNameError, setFirstNameError] = useState('')
  const [lastNameError, setLastNameError] = useState('')

  const { userLoggedIn } = useAuth()

  const togglePasswordVisibility1 = () => setShowPassword1(!showPassword1)
  const togglePasswordVisibility2 = () => setShowPassword2(!showPassword2)

  const handleInputChange = (e) => {
    const { id, value } = e.target

    switch (id) {
      case 'first-name':
        setFirstName(value)
        break
      case 'last-name':
        setLastName(value)
        break
      case 'email':
        setEmail(value)
        break
      case 'password':
        setPassword(value)
        break
      case 'confirm-password':
        setConfirmPassword(value)
        break
      default:
        break
    }
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    if (password !== confirmPassword) {
      setErrorMessage('Passwords do not match')
      return
    }

    if (!isRegistering) {
      setIsRegistering(true)
      try {
        await doCreateUserWithEmailAndPassword(email, password)
        navigate('/home')
      } catch (error) {
        setErrorMessage(error.message)
        setIsRegistering(false)
      }
    }
  }

  return (
    <>
      {userLoggedIn && (<Navigate to={'/home'} replace={true} />)}

      <div className='signup-container'>
        <form className='signup-form' onSubmit={onSubmit}>
          <h5>Create a New Account</h5>
          <br />
          {/*<div className='input-wrapper'>
            <FaUser className='input-icon' />
            <input
              type='text'
              id='first-name'
              name='first-name'
              placeholder='First Name'
              value={firstName}
              onChange={handleInputChange}
              autoComplete='given-name'
              required
            />
          </div>
          <div className='input-wrapper'>
            <FaUser className='input-icon' />
            <input
              type='text'
              id='last-name'
              name='last-name'
              placeholder='Last Name'
              value={lastName}
              onChange={handleInputChange}
              autoComplete='family-name'
              required
            />
          </div>
          */}
          <div className='input-wrapper'>
            <FaUser className='input-icon' />
            <input
              type='text'
              id='first-name'
              name='first-name'
              placeholder='First name'
              value={firstName}
              onChange={handleInputChange}
            />
            {firstNameError && (
              <div className='error-message'>
                <FaExclamationCircle className='error-icon' />
                {firstNameError}
              </div>
            )}
          </div>
          <div className='input-wrapper'>
            <FaUser className='input-icon' />
            <input
              type='text'
              id='last-name'
              name='last-name'
              placeholder='Last name'
              value={lastName}
              onChange={handleInputChange}
            />
            {lastNameError && (
              <div className='error-message'>
                <FaExclamationCircle className='error-icon' />
                {lastNameError}
              </div>
            )}
          </div>

          <div className='input-wrapper'>
            <FaEnvelope className='input-icon' />
            <input
              type='email'
              id='email'
              name='email'
              placeholder='Email'
              value={email}
              onChange={handleInputChange}
              autoComplete='email'
              required
            />
          </div>
          <div className='form-input'>
            <div className='input-wrapper'>
              <FaLock className='input-icon' />
              <input
                type={showPassword1 ? 'text' : 'password'}
                id='password'
                name='password'
                placeholder='Password'
                value={password}
                onChange={handleInputChange}
                autoComplete='new-password'
                required
                disabled={isRegistering}
              />
              {showPassword1 ? (
                <FaEyeSlash
                  className='eye-icon'
                  onClick={togglePasswordVisibility1}
                />
              ) : (
                <FaEye className='eye-icon' onClick={togglePasswordVisibility1} />
              )}
            </div>
            <div className='input-wrapper'>
              <FaLock className='input-icon' />
              <input
                type={showPassword2 ? 'text' : 'password'}
                id='confirm-password'
                name='confirm-password'
                placeholder='Confirm Password'
                value={confirmPassword}
                onChange={handleInputChange}
                autoComplete='off'
                required
                disabled={isRegistering}
              />
              {showPassword2 ? (
                <FaEyeSlash
                  className='eye-icon'
                  onClick={togglePasswordVisibility2}
                />
              ) : (
                <FaEye className='eye-icon' onClick={togglePasswordVisibility2} />
              )}
            </div>
          </div>
          {errorMessage && (
            <div className='error-message'>
              <FaExclamationCircle className='error-icon' />
              {errorMessage}
            </div>
          )}
          <button
            type='submit'
            disabled={isRegistering}
            className={`w-full px-4 py-2 text-white font-medium rounded-lg ${isRegistering ? 'bg-gray-300 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-xl transition duration-300'}`}
          >
            {isRegistering ? 'Signing Up...' : 'Sign Up'}
          </button>
          <br />
          <h6>
            Already have an Account?&nbsp;
            <Link to='/login'>Login</Link>
          </h6>
        </form>
      </div>
    </>
  )
}

export default SignUp
