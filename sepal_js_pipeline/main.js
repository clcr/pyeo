// use https://developers.google.com/earth-engine/apidocs for documentation
require("dotenv").config();
const fs = require("fs")
const ee = require("@google/earthengine");

// import custom modules
const inputs = require("./config/inputs");
const imagery = require("./src/imagery");
// classification module would go here
// change detection module would go here

// auth
const keyPath = process.env.EE_PRIVATE_KEY_PATH;
const privateKey = JSON.parse(fs.readFileSync(keyPath, "utf8"));

// run main, wrapped within an authentication call
ee.data.authenticateViaPrivateKey(privateKey, () => {
    ee.initialize(null, null, () => {
        console.log("Earth Engine initialised. Starting pipeline...");

        // load user-inputs
        const params = inputs.getParameters();

        // get baseline mosaic
        console.log("Generating baseline mosaic...")
        const baseline = imagery.getBaselineMosaic(
            params.roi,
            params.baseline.start,
            params.baseline.end,
            params.cloudCoverThreshold
        );
        //console.log("Baseline band names : ", baseline.bandNames().getInfo());

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
        
    }, (e) => console.error("Initialisation error: ", e));
}, (e) => console.error("Authentication error: ", e));