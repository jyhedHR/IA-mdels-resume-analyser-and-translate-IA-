const axios = require('axios');

// Function to analyze CV text
const analyzeCV = async (cvText) => {
  try {
    const response = await axios.post('http://localhost:5000/analyze', {
      cvText: cvText
    });
    
    console.log(response.data);  // Process the entities returned
  } catch (error) {
    console.error('Error analyzing CV:', error);
  }
};

module.exports = { analyzeCV };