// needed to allow this file to be required by main.js
const ee = require('@google/earthengine');

// cloud-masking function
function maskS2clouds(image) {
    const qa = image.select('QA60');
  
    // bits 10 and 11 are clouds and cirrus, respectively.
    const cloudBitMask = 1 << 10;
    const cirrusBitMask = 1 << 11;
    
    // both flags should be set to zero, indicating clear conditions.
    const mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
        
    // apply the mask and scale the reflectance values to standard 0-1 range
    return image.updateMask(mask).divide(10000)
        .copyProperties(image) // copies all non-system properties
        .copyProperties(image, ["system:time_start",
            "system:index"]
    )
};

// calculating NDVI
function addNDVI(image) {
    const ndvi = image.normalizedDifference(["B5", "B4"]).rename("NDVI");
    return image.addBands(ndvi);
};

// generates a cloud free composite for a given time period and Region of Interest (ROI)
function getBaselineMosaic(roi, startDate, endDate, cloudCoverThreshold, bandsOfInterest) {
    const collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(startDate, endDate)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudCoverThreshold)) // filter very cloudy scenes
        .map(maskS2clouds)
        .map(addNDVI)
        .map((image) => { // get metadata for every image for date reporting
            return image.select(bandsOfInterest)
                .copyProperties(image)
                .copyProperties(image, ["system:time_start",
                    "system:index"
                ])
            })
        ;

    // report mosaic metadata
    const count = collection.size();
    const dates = collection.aggregate_array("system:time_start").map(function(time) {
        return ee.Date(time).format("YYYY-MM-dd");
    });

    // evaluate fetches information from GEE servers without preventing the rest of the pipeline from computing, which getInfo() would do
    count.evaluate((n) => console.log(`Mosaic consists of ${n} images.`));
    dates.evaluate((dates) => console.log(`Image dates: ${dates.join(', ')}`));

    return collection.median().clip(roi)
}

function getChangeTimeSeries({roi, startDate, endDate, cloudCoverThreshold, bandsOfInterest}) {
    const collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(startDate, endDate)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudCoverThreshold)) // filter very cloudy scenes
        .map(maskS2clouds)
        .map(addNDVI)
        .map((image) => {
            return image.select(bandsOfInterest)
                .copyProperties(image)
                .copyProperties(image, ["system:time_start",
                    "system:index"])
                });
    
    // report timeseries metadata
    const count = collection.size();
    const dates = collection.aggregate_array("system:time_start").map(function(time) {
        return ee.Date(time).format("YYYY-MM-dd");
    });

    // evaluate fetches information from GEE servers without preventing the rest of the pipeline from computing, which getInfo() would do
    count.evaluate((n) => console.log(`\nChange Timeseries consists of ${n} images.`));
    dates.evaluate((dates) => console.log(`\nChange Timeseries image dates: ${dates.join(', ')}`));

    return collection.map((image) => {
        return image.clip(roi)
            .copyProperties(image)
            .copyProperties(image, ["system:time_start",
                "system:index"])
            });
}

// export the function to be required by main.js
module.exports = {
    getBaselineMosaic,
    getChangeTimeSeries
};