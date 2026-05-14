// use https://developers.google.com/earth-engine/apidocs for documentation
require("dotenv").config();
const fs = require("fs")
const ee = require("@google/earthengine");

// import custom modules
const inputs = require("./config/inputs");
const imagery = require("./src/imagery");
const classification = require("./src/classification");
const changeDetection = require("./src/changeDetection");

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
        const baseline = imagery.getBaselineMosaic(
            params.roi,
            params.baseline.start,
            params.baseline.end,
            params.cloudCoverThreshold,
            params.bandsOfInterest
        );

        baseline.bandNames().evaluate((bands, error) => {
            if (error) console.error('Band error:', error);
            else console.log('\nBaseline image bands:', bands);
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
            console.log("\nBaseline Mosaic Quicklook:", url);
        });

        /////////
        // create a classifier on the baseline
        /////////
        const modelResults = classification.trainGBClassifier({
            baselineImage: baseline,
            trainingPoints: params.trainingFeatures,
            classProperty: "class"
        });

        // use evaluate to get model performance information
        modelResults.trainingAccuracy.evaluate((acc) => {
            console.log(`\nTraining accuracy: ${(acc * 100).toFixed(2)}%`);
        });
        modelResults.validationAccuracy.evaluate((acc) => {
            console.log(`\nValidation accuracy: ${(acc * 100).toFixed(2)}%`);
        });

        /////////
        // apply the classifier on the baseline
        /////////
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
            console.log("\nClassified Baseline Quicklook:", url);
        });

        /////////
        // get a cloud masked change timeseries
        /////////
        const timeSeries = imagery.getChangeTimeSeries({
            roi: params.roi,
            startDate: params.change.start,
            endDate: params.change.end,
            cloudCoverThreshold: params.cloudCoverThreshold,
            bandsOfInterest: params.bandsOfInterest
        })

        const logFirstImageMetadata = (collection) => {
        const firstImage = collection.first();
        
        firstImage.toDictionary().evaluate((props, error) => {
            if (error) {
                console.error("Error fetching image properties:", error);
            } else {
                console.log("\n--- First Image Metadata ---");
                console.log(`ID: ${props['PRODUCT_ID']}`);
                console.log(`Cloud Cover (Entire Tile): ${props['CLOUDY_PIXEL_PERCENTAGE'].toFixed(2)}%`);
                console.log(`Processing Baseline: ${props['PROCESSING_BASELINE']}`);
            }
        });

        //////
        // get all of image metadata if further debugging needed
        //

        // firstImage.propertyNames().evaluate((properties, error) => {
        //     if (error) {
        //         console.error("error", error);
        //     } else {
        //         console.log(`${properties}`)
        //     }
        // })
        };

        logFirstImageMetadata(timeSeries);

        /////////
        // apply the classifier on the change timeseries
        /////////
        const classifiedTimeseries = classification.classifyTimeseries({
            collection: timeSeries,
            trainedClassifier: modelResults.classifier
        })

        classifiedTimeseries.first().getThumbURL(classifiedVisParams, (url) => {
            console.log("\nClassified Timeseries 1st image Quicklook:", url);
        });

        
    }, (e) => console.error("Initialisation error: ", e));
}, (e) => console.error("Authentication error: ", e));