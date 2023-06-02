const express = require("express")
const path = require("path")

const app = express();
const port = 5000;
const hostname = "127.0.0.1";

app.use('/results/', express.static(path.resolve(__dirname, "frontend", "results_tool")))
app.use('/preprocessing/', express.static(path.resolve(__dirname, "frontend", "preprocessing_tool")))


app.get('/', (req, res) => {
    try {
        res.sendFile(path.resolve(__dirname, 'frontend', 'index.html'));
    } catch (err) {
        res.status(500).json({message: err.message})
    }
});

app.get('/results', (req, res) => {
    try {
        res.sendFile(path.resolve(__dirname, 'frontend', 'results_tool', 'index.html'));
    } catch (err) {
        res.status(500).json({message: err.message})
    }
})

app.get('/preprocessing', (req, res) => {
    try {
        res.sendFile(path.resolve(__dirname, 'frontend', 'preprocessing_tool', 'index.html'));
    } catch (err) {
        res.status(500).json({message: err.message})
    }
})


app.listen(port, hostname, 511, () => {
    console.log(`Serving app at http://${hostname}:${port}`);
});