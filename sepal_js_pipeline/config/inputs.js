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
            end: "2026-05-14"
        },

        bandsOfInterest: ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI'],

        cloudCoverThreshold: 30,

        deltaNDVI: 0.2, // default in pyeo

        changeFromClass: 0,

        changeToClass: 1,

        minConsecutiveChanges: 2, // or 5

        classLabels: {
            "forest": 0,
            "grassland": 1,
            "bareSoil": 2,
            //"agriculture": 2,
            "urban": 3,
            "water": 4
        },

        classColours: {
            "forest": "006400",      // darker green
            "grassland": "7CFC00",   // lighter green
            "bareSoil": "8B4513",    // muddy brown
            //"agriculture": "FFFF00", // yellow
            "urban": "808080",       // grey
            "water": "0000FF"        // blue
        }
    };
}

// enable getParameters to be accessible when required
module.exports = {
    getParameters
}