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
        console.log("1. Earth Engine initialised. Starting pipeline...");

        // load user-inputs
        const params = inputs.getParameters();

        /////////
        // get baseline mosaic
        /////////
        console.log("2. Generating baseline mosaic...")
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
            console.log("\n Baseline Mosaic Quicklook:", url);
        });

        /////////
        // create a classifier on the baseline
        /////////
        console.log("3. Training a Classifier...");
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

        /////////
        // apply the classifier on the baseline
        /////////
        console.log("4. Applying the Classifier on the baseline...");
        const classifiedBaseline = classification.classifyImage({
            image: baseline,
            trainedClassifier: modelResults.classifier
        })

        // get the parameters for the classification quicklook
        const classValues = Object.values(params.classLabels);
        const maxClassValue = Math.max(...classValues);
        const dynamicPalette = Object.keys(params.classLabels)
            .sort((a, b) => params.classLabels[a] - params.classLabels[b])
            .map(key => params.classColours[key]);

        // quicklook of the classified baseline
        const classifiedVisParams = {
            bands: ["classification"],
            min: 0,
            max: maxClassValue,
            palette: dynamicPalette,
            dimensions: 800
        };
        classifiedBaseline.getThumbURL(classifiedVisParams, (url) => {
            console.log("\n Classified Baseline Quicklook:", url);
        });

        /////////
        // get a cloud masked change timeseries
        /////////
        console.log("5. Getting a Timeseries of Change Images...");
        const timeSeries = imagery.getChangeTimeSeries({
            roi: params.roi,
            startDate: params.change.start,
            endDate: params.change.end,
            cloudCoverThreshold: params.cloudCoverThreshold,
            bandsOfInterest: params.bandsOfInterest
        })

        /////////
        // apply the classifier on the change timeseries
        /////////
        // console.log("6. Applying the Classifier on the Timeseries...");
        // const classifiedTimeseries = classification
        
    }, (e) => console.error("Initialisation error: ", e));
}, (e) => console.error("Authentication error: ", e));