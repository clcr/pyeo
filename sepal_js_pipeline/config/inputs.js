// needed to allow this file to be required by main.js
const ee = require('@google/earthengine');

// define a constant to export
const getParameters = () => {
    // declare asset paths
    const shimbaHills = ee.FeatureCollection("projects/ee-matthewpayne/assets/A4F/shimba_hills");
    const trainingFeatures = ee.FeatureCollection("projects/ee-matthewpayne/assets/A4F/shimba_hills_20250119_trainingFeatures");

    return {
        // define nested objects for cleaner reading
        roi: shimbaHills.geometry(),

        trainingFeatures: trainingFeatures,

        baseline: {
            start: "2025-01-01",
            end: "2025-12-31"
        },

        change: {
            start: "2026-01-01",
            end: "2026-05-08"
        },

        cloudCoverThreshold: 25,

        changeFromClass: 1,

        changeToClass: 3,

        classLabels: {
            "forest": 1,
            "bareSoil": 2,
            "agriculture": 3,
            "grassland": 4,
            "urban": 5,
            "water": 6
        }
    };
}

// enable getParameters to be accessible when required
module.exports = {
    getParameters
}