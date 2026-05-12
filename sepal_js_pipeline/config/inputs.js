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

        bandsOfInterest: ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],

        cloudCoverThreshold: 25,

        changeFromClass: 0,

        changeToClass: 1,

        classLabels: {
            "forest": 0,
            "bareSoil": 1,
            "agriculture": 2,
            "grassland": 3,
            "urban": 4,
            "water": 5
        },

        classColours: {
            "forest": "006400",      // darker green
            "bareSoil": "8B4513",    // muddy brown
            "agriculture": "FFFF00", // yellow
            "grassland": "7CFC00",   // lighter green
            "urban": "808080",       // grey
            "water": "0000FF"        // blue
        }
    };
}

// enable getParameters to be accessible when required
module.exports = {
    getParameters
}