import React, { useState } from 'react';
import '../styles/MinimalistUI.css';

const MinimalistUI = () => {
  const [userInput, setUserInput] = useState('');
  const [response, setResponse] = useState('');

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  const handleSubmit = () => {
    // Implement your logic to generate a response based on the user input
    const generatedResponse = `You entered: ${userInput}`;
    setResponse(generatedResponse);
  };

  return (
    <div className="minimalist-ui">
      <div className="input-container">
        <input
          type="text"
          value={userInput}
          onChange={handleInputChange}
          placeholder="Enter your query..."
        />
        <button onClick={handleSubmit}>Submit</button>
      </div>
      <div className="response-container">
        <p>{response}</p>
      </div>
    </div>
  );
};

export default MinimalistUI;