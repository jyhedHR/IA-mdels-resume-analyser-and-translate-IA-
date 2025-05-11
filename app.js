import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [cvText, setCvText] = useState('');
  const [entities, setEntities] = useState([]);
  
  const handleChange = (event) => {
    setCvText(event.target.value);
  };
  
  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://localhost:5000/analyze', { cvText });
      setEntities(response.data.entities);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <h1>CV Analysis</h1>
      <textarea value={cvText} onChange={handleChange} rows="10" cols="50" />
      <button onClick={handleSubmit}>Analyze</button>

      <div>
        <h2>Named Entities:</h2>
        <ul>
          {entities.map((ent, index) => (
            <li key={index}>{ent.text} ({ent.label})</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
