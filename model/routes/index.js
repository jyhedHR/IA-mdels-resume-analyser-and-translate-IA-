var express = require('express');
var router = express.Router();
const { analyzeCV } = require('../controller/ia.js');
/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});
// POST route to analyze CV
router.post('/analyse', analyzeCV);

module.exports = router;
