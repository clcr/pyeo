// needed to allow this file to be required by main.js
const ee = require('@google/earthengine');

/** 
 * This function samples an image using training points and trains a gradient-boosted classifier
 * @param {ee.Image} baselineImage - The baseline image to sample spectral values from
 * @param {ee.FeatureCollection} trainingPoints - the FeatureCollection containing the point geometries which overlap baselineImage
 * @param {string} classProperty - the name of the column containing the integer class labels, denoting each land cover class
 * @returns {Object} - an object containing a trained Earth Engine classifier, training accuracy and validation accuracy metrics
*/
function trainGBClassifier({baselineImage, trainingPoints, classProperty}) {

    /**
     * helper function that subtracts 1 from the class property to ensure 0-indexing for gradientBoosted 
     * * @param {ee.Feature} feature - A single feature from the training points collection.
     * @returns {ee.Feature} The feature with its class property decremented by 1.
     */
    const remapForZeroIndexed = (feature) => {
        const currentClass = ee.Number(feature.get(classProperty));
        return feature.set(classProperty, currentClass.subtract(1));
    };

    // ensure zero-indexing for gradientBoosted
    zeroIndexedPoints = trainingPoints.map(remapForZeroIndexed);

    // sample the provided image at the coordinates specified by trainingPoints
    const sampleFeatures = baselineImage.sampleRegions({
        // https://developers.google.com/earth-engine/apidocs/ee-image-sampleregions
        collection: trainingPoints,
        properties: [classProperty],
        scale: 10,
        tileScale: 4
    });

    // add a random column and use this to split 75% for training, 25% for validation
    const withRandom = sampleFeatures.randomColumn("random");
    const trainingSample = withRandom.filter(ee.Filter.lte("random", 0.75));
    const validationSample = withRandom.filter(ee.Filter.gt("random", 0.75));

    // report if any classes are not represented
    trainingSample.aggregate_histogram(classProperty).evaluate((hist => {console.log(`Training class distribution : ${hist}`)}));

    // train the classifier
    const trainedClassifier = ee.Classifier.smileGradientTreeBoost({
        numberOfTrees: 50,
        seed: 42
    }).train({
        // https://developers.google.com/earth-engine/apidocs/ee-classifier-smilegradienttreeboost
        features: trainingSample,
        classProperty: classProperty,
        inputProperties: baselineImage.bandNames()
    });

    // evaluate training performance
    const trainingAccuracy = trainedClassifier.confusionMatrix().accuracy()

    // evaluate performance on unseen validation data
    const validatedSample = validationSample.classify(trainedClassifier);
    
    const validationAccuracy = validatedSample.errorMatrix(classProperty, "classification").accuracy();

    // return
    return {
        classifier: trainedClassifier,
        trainingAccuracy: trainingAccuracy,
        validationAccuracy: validationAccuracy
    };
}

module.exports = {
    trainGBClassifier//,
    // classifyImage,
    // classifyTimeseries
};