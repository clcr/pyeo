// needed to allow this file to be required by main.js
const ee = require('@google/earthengine');

// cloud-masking function
function maskS2clouds(image) {
  var qa = image.select('QA60');
  
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
    
  // Apply the mask and scale the reflectance values to standard 0-1 range
  return image.updateMask(mask).divide(10000)
      .copyProperties(image, ["system:time_start"]);
}

// generates a cloud free composite for a given time period and Region of Interest (ROI)
function getBaselineMosaic(roi, startDate, endDate, cloudCoverThreshold) {
    const collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(startDate, endDate)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudCoverThreshold)) // filter very cloudy scenes
        .map(maskS2clouds)
    return collection.median().clip(roi)
}

// export the function to be required by main.js
module.exports = {
    getBaselineMosaic
};