import React, { useRef } from 'react'
import './Home.css'
// import background2 from '../assets/background2.MOV'
import Header from '../header/index'

const Home = () => {
  const fileInputRef = useRef(null)

  const handleIdentifyClick = () => {
    alert('Identifying the apple... hang on..')
  }

  const handleUploadClick = () => {
    fileInputRef.current.click()
  }

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    if (file) {
      console.log('File selected:', file.name)
      // Handle the file upload here (e.g., send it to the server or perform some action)
    }
  }

  return (
    <div className='home-container'>
     <Header />
      {/* <video className='backgroundVideo' autoPlay loop muted>
        <source src={background2} type='video/mp4' />
      </video> */}
      <div className='content'>
        <h1 className='title'>Apple Variety Identification Application</h1>
        <h5 className='subtitle'>Apples Unveiled: Identify with Ease</h5>
        <div className='buttons'>
          <button className='button' onClick={handleUploadClick}>
            Upload Image
          </button>
          <button className='button primary' onClick={handleIdentifyClick}>
            Identify
          </button>
          <input
            type='file'
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
        </div>
      </div>
    </div>
  )
}

export default Home