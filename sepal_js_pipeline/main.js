// use https://developers.google.com/earth-engine/apidocs for documentation
require("dotenv").config();
const fs = require("fs")
const ee = require("@google/earthengine");

// import custom modules
const inputs = require("./config/inputs");
const imagery = require("./src/imagery");
const classification = require("./src/classification");
// change detection module require goes here

// auth
const keyPath = process.env.EE_PRIVATE_KEY_PATH;
const privateKey = JSON.parse(fs.readFileSync(keyPath, "utf8"));

// run main, wrapped within an authentication call
ee.data.authenticateViaPrivateKey(privateKey, () => {
    ee.initialize(null, null, () => {
        console.log("Earth Engine initialised. Starting pipeline...");

        // load user-inputs
        const params = inputs.getParameters();

        /////////
        // get baseline mosaic
        /////////
        console.log("Generating baseline mosaic...")
        const baseline = imagery.getBaselineMosaic(
            params.roi,
            params.baseline.start,
            params.baseline.end,
            params.cloudCoverThreshold,
            params.bandsOfInterest
        );

        baseline.bandNames().evaluate((bands, error) => {
            if (error) console.error('Band error:', error);
            else console.log('Baseline image bands:', bands);
        });

        // quicklook of the baseline
        const visParams = {
            bands: ["B4", "B3", "B2"],
            min: 0,
            max: 0.3,
            gamma: 1.4,
            dimensions: 800
        };
        baseline.getThumbURL(visParams, (url) => {
            console.log("\n Quicklook:", url);
        });

        /////////
        // create a classifier on the baseline
        /////////
        console.log("Training a Classifier...");
        const modelResults = classification.trainGBClassifier({
            baselineImage: baseline,
            trainingPoints: params.trainingFeatures,
            classProperty: "class"
        });

        // use evaluate to get model performance information
        modelResults.trainingAccuracy.evaluate((acc) => {
            console.log(`Training accuracy: ${(acc * 100).toFixed(2)}%`);
        });
        modelResults.validationAccuracy.evaluate((acc) => {
            console.log(`Validation accuracy: ${(acc * 100).toFixed(2)}%`);
        });
        
    }, (e) => console.error("Initialisation error: ", e));
}, (e) => console.error("Authentication error: ", e));