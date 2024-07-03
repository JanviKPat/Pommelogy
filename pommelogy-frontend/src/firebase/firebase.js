import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyBHJijMCGvN9k3ucT16AG1IUxm_ohJkSFw",
  authDomain: "pommelogy-3c333.firebaseapp.com",
  projectId: "pommelogy-3c333",
  storageBucket: "pommelogy-3c333.appspot.com",
  messagingSenderId: "176353550025",
  appId: "1:176353550025:web:11e767e33e95bc4db43718",
  measurementId: "G-WVREM0PDKH"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app)



export { app, auth };
