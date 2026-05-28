var pyeo = require('users/matthewjpayne1/a4f:pyeoChangeAlerts')

// AOI: ~15 km box over Mato Grosso, Brazil.
var aoi = ee.Geometry.Rectangle([-55.30, -11.65, -55.15, -11.50])//.buffer(5000)
Map.centerObject(aoi, 12)

// Baseline = median over Jan-Mar 2022. Monitoring = individual S2 acquisitions
// over Apr-Dec 2022 (no compositing — preserves temporal granularity).
var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
var cloudThreshold = 50
var FOREST = 1
var SOIL = 2
var CROPS = 3
var changeFromClasses = [FOREST]
var changeToClasses = [SOIL, CROPS]
var allClasses = [FOREST, SOIL, CROPS]

// **********
// visualisation parameters
var classColourMap = {
  1: "green", // forest
  2: "yellow", // soil
  3: "pink" // crops
}
// dynamic from and to palettes
var dynamicFromPalette = changeFromClasses.map(function(classId) {
  return classColourMap[classId];
})
var dynamicToPalette = changeToClasses.map(function(classId) {
  return classColourMap[classId];
})
var dynamicFullPalette = allClasses.map(function(classId) {
  return classColourMap[classId];
})

var visClassParams = {
  min: Math.min.apply(null, allClasses),
  max: Math.max.apply(null, allClasses),
  palette: dynamicFullPalette
};

var fromClassParams = {
  min: Math.min.apply(null, changeFromClasses),
  max: Math.max.apply(null, changeFromClasses),
  palette: dynamicFromPalette 
};

var toClassParams = {
  min: Math.min.apply(null, changeToClasses),
  max: Math.max.apply(null, changeToClasses),
  palette: dynamicToPalette 
};

var ndviParams = {
  min: -0.2,
  max: 1,
  palette: ["white", "green"] // specifies the upper and lower range
}
// **********

/**
 * Joins the S2 cloud probability collection to a given S2 SR collection and masks clouds.
 *
 * @param {ee.ImageCollection} srCol - The input Sentinel-2 Surface Reflectance collection.
 * @return {ee.ImageCollection} The cloud-masked and scaled Sentinel-2 collection.
 */

// var maskS2clouds = function (img) {
//     var qa = img.select('QA60')
//     var mask = qa.bitwiseAnd(1 << 10).eq(0) // 0 = no clouds
//         .and(qa.bitwiseAnd(1 << 11).eq(0)) // 0 = no cirrus
//     return img.updateMask(mask)
//         .divide(10000)
//         .copyProperties(img).copyProperties(img, ['system:time_start'])
// }

var applyS2Cloudless = function(srCol, cloudThreshold) {
    // Load the s2cloudless collection, filtering to your AOI
    var s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi);
    
    // Define an Inner Join to match the SR images with their cloud probability counterparts
    var join = ee.Join.saveFirst('s2cloudless');
    var condition = ee.Filter.equals({
        leftField: 'system:index',
        rightField: 'system:index'
    });
    
    // Apply the join
    var joinedCol = ee.ImageCollection(join.apply(srCol, s2Clouds, condition));
    
    // Map over the joined collection to apply the mask
    return joinedCol.map(function(img) {
        // Extract the probability image from the joined property
        var prob = ee.Image(img.get('s2cloudless')).select('probability');
        
        // Create a mask where cloud probability is below a threshold (50% is a good baseline)
        var isNotCloud = prob.lt(cloudThreshold); 
        
        // Apply the mask, scale the optical bands, and preserve the time property
        return img.updateMask(isNotCloud)
            .divide(10000)
            .copyProperties(img, ['system:time_start']);
    });
};

var addNdvi = function (img) {
    return img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
}

var prep = function (col) {
    //return col.map(maskS2clouds).map(addNdvi).select(BANDS.concat(['NDVI']))
    return applyS2Cloudless(col, cloudThreshold).map(addNdvi).select(BANDS.concat(['NDVI']))
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))

var baselineImage = prep(
    s2.filterDate('2022-01-01', '2022-06-01') // change from 2022-04-01
).median().clip(aoi)

var monitoringImages = prep(
    s2.filterDate('2022-06-01', '2023-01-01') // change from 2022-04-01
).sort('system:time_start')
    .map(function (img) { return img.clip(aoi) })

var imageList = monitoringImages.toList(38)
var secondImage = ee.Image(imageList.get(1))

// Map.addLayer(
//   monitoringImages.first(),
//   {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.5, gamma: 1.4},
//   'First monitoring acquisition'
// )

// Inline training: forest / non-forest points within the AOI. Test fixture
// only — disappears once the SEPAL CLASSIFICATION recipe wrapper is in place.
var trainingPoints = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([-55.283, -11.560]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-55.270, -11.585]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-55.255, -11.610]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-55.230, -11.555]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-55.270, -11.620]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-55.213, -11.550]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-55.175, -11.580]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-55.264, -11.504]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-55.184, -11.587]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-55.185, -11.588]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-55.161, -11.625]), {'class': CROPS}),
    ee.Feature(ee.Geometry.Point([-55.215, -11.530]), {'class': CROPS}),
    ee.Feature(ee.Geometry.Point([-55.283, -11.632]), {'class': CROPS}),
    ee.Feature(ee.Geometry.Point([-55.165, -11.644]), {'class': CROPS}),
    ee.Feature(ee.Geometry.Point([-55.169, -11.585]), {'class': CROPS})
])

var trainingSamples = baselineImage.sampleRegions({
    collection: trainingPoints,
    properties: ['class'],
    scale: 10
})

var classifier = ee.Classifier.smileRandomForest(50).train({
    features: trainingSamples,
    classProperty: 'class',
    inputProperties: BANDS.concat(['NDVI'])
})

// Pre-classify to match the contract runPyeoChangeAlerts expects.
var classifiedBaselineImage = baselineImage
    .classify(classifier).rename('classification')
    .addBands(baselineImage.select('NDVI'))
    .clip(aoi)

var classifiedMonitoringCollection = monitoringImages.map(function (img) {
    return img.classify(classifier).rename('classification')
        .addBands(img.select('NDVI'))
        .copyProperties(img, ['system:time_start'])
})


var alerts = pyeo.run_change_detection({
    aoi: aoi,
    classifiedBaseline: classifiedBaselineImage,
    classifiedMonitoringCollection: classifiedMonitoringCollection,
    changeFromClasses: changeFromClasses,
    changeToClasses: changeToClasses,
    minConsecutiveDetections: 2,
    dNdviGate: {band: 'NDVI', threshold: 0.20}
})

var imageCountVis = {
  min: 0,
  max: 38,
  palette: [
    '#d7191c', // Red: Very few valid images
    '#fdae61', // Orange
    '#ffffbf', // Yellow: Moderate availability
    '#a6d96a', // Light Green
    '#1a9641'  // Dark Green: Excellent availability
  ] 
};

var occludivityVis = {
  min: 0,
  max: 38,
  palette: [
    '#1a9641', // Dark Green: Rarely occluded (Clear skies)
    '#a6d96a', // Light Green
    '#ffffbf', // Yellow: Moderately occluded
    '#fdae61', // Orange
    '#d7191c'  // Red: Highly occluded (Persistent clouds)
  ]
};

Map.addLayer(
  alerts.fromClassCollection.first(),
  fromClassParams,
  "First image of the fromClassCollection"
)

Map.addLayer(
  alerts.toClassCollection.first(),
  toClassParams,
  "First image of the toClassCollection"
)

Map.addLayer(
  alerts.changeEvents.first().select("delta_ndvi"),
  ndviParams,
  "Delta NDVI of the first monitoring image"
)

Map.addLayer(
  alerts.changeEvents.first().select("delta_ndvi_thresholded"),
  ndviParams,
  "Delta NDVI thresholded >=0.2 of the first monitoring image"
)

// Map.addLayer(
//   alerts.changeReport.select('valid_image_count'),
//   imageCountVis,
//   'Available Image Count', false
// );

// Map.addLayer(
//   alerts.changeReport.select('occluded_count'),
//   occludivityVis,
//   'Occluded Pixel Count', false
// );

Map.addLayer(
  classifiedMonitoringCollection.first().select("classification"),
  visClassParams,
  'First monitoring acquisition - CLASSIFIED'
)

Map.addLayer(
  classifiedBaselineImage.select('classification'),
  visClassParams,
  'Baseline class map'
)

// var point = ee.Geometry.Point([-55.1514, -11.5683]);
// var changeReportAtPoint = alerts.changeReport.reduceRegion({
//   reducer: ee.Reducer.first(),
//   geometry: point,
//   scale: 10
// })

// print(changeReportAtPoint)

// changeReportAtPoint.evaluate(function(result) {
//   if (result.first_date > 0) {
//     // use JS to create a Date object, which has .toUTCString()
//     // Date is an EE function that returns a string, strings don't have .toUTCString()
//     var readableFirstChange = new Date(result.first_date).toUTCString();
//     var readableLastChange = new Date(result.last_date).toUTCString();
    
//     print("First change was on: ", readableFirstChange);
//     print("Most recent change on: ", readableLastChange);
//   }
//   else {
//     print("No change at this location")
//   }
// })