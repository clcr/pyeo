var pyeo = require('users/matthewjpayne1/a4f:pyeoChangeAlerts')
var cloudMasking = require('users/matthewjpayne1/a4f:cloudMasking');
// ==============================================================================
// 1. PARAMETERS & CONSTANTS
// ==============================================================================

// baseline = median over Jan-Mar 2022. Monitoring = individual S2 acquisitions
// over Apr-Dec 2022 (no compositing — preserves temporal granularity).
// AOI: ~15 km box over Mato Grosso, Brazil.
var aoi = ee.Geometry.Rectangle([-55.30, -11.65, -55.15, -11.50]);

var BASELINE_START = '2022-01-01';
var BASELINE_END = '2022-06-01';
var MONITORING_START = '2022-06-01';
var MONITORING_END = '2023-01-01';

var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
var MAX_CLOUD_PERCENTAGE = 30;
var MAX_CLOUD_THRESHOLD_PER_PIXEL = 50;

var FOREST = 1;
var SOIL = 2;
var CROPS = 3;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, CROPS];
var allClasses = [FOREST, SOIL, CROPS];

// parameter objects for imagery acquisition and cloud masking
var baselineParams = {
  aoi: aoi,
  startDate: BASELINE_START,
  endDate: BASELINE_END,
  useSR: true,
  maxCloudProbability: MAX_CLOUD_THRESHOLD_PER_PIXEL,
  method: "BOTH"
}

var monitoringParams = {
  aoi: aoi,
  startDate:  MONITORING_START,
  endDate: MONITORING_END,
  useSR: true,
  maxCloudProbability: MAX_CLOUD_THRESHOLD_PER_PIXEL,
  method: "BOTH"
}

// ==============================================================================
// 2. MAP INITIALISATION
// ==============================================================================
Map.centerObject(aoi, 12)
Map.addLayer(aoi, {color: 'red'}, 'AOI Outline', false);

// ==============================================================================
// 3. VISUALISATION PARAMETERS
// ==============================================================================

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

// ==============================================================================
// 4. HELPER FUNCTIONS
// ==============================================================================

/**
 * Joins the S2 cloud probability collection to a given S2 SR collection and masks clouds.
 *
 * @param {ee.ImageCollection} srCol - The input Sentinel-2 Surface Reflectance collection.
 * @return {ee.ImageCollection} The cloud-masked and scaled Sentinel-2 collection.
 */
var applyS2Cloudless = function(srCol, cloudThreshold) {
    // Load the s2cloudless collection, filtering to AOI
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
        
        // Create a mask where cloud probability is below a threshold
        var isNotCloud = prob.lt(cloudThreshold); 
        
        // Apply the mask, scale the optical bands, and preserve the time property
        return img.updateMask(isNotCloud)
            .divide(10000)
            .copyProperties(img, ['system:time_start']);
    });
};


// NDVI calculation
var addNDVI = function (img) {
    return img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'));
};

// prepare a base collection for baseline and monitoring images
var prep = function (col) {
    return applyS2Cloudless(col, MAX_CLOUD_THRESHOLD_PER_PIXEL)
      .map(addNDVI)
      .select(BANDS.concat(['NDVI']));
};

// group images by date and mosaics overlapping tiles from the same orbit pass
var dailyMosaic = function(col) {
  // add a date string to each image
  var colWithDate = col.map(function(image) {
    var date = image.date().format("YYYY-MM-dd");
    return image.set("date_str", date);
  })
  // get a list of the unique dates in a collection
  var distinctDates = colWithDate.distinct("date_str").aggregate_array("date_str");

  // map over the distinct dates to create a mosaic per day
  var mosaicedImages = distinctDates.map(function(date) {
    var dailyCol = colWithDate.filter(ee.Filter.equals("date_str", date));

    // get the first image to keep metadata
    var firstImage = dailyCol.first();

    // mosaic out the tile overlap https://developers.google.com/earth-engine/apidocs/ee-imagecollection-mosaic
    // ee.ImageCollection.mosaic composites images according to their position in
    // the collection (priority is last to first) and pixel mask status, where
    // invalid (mask value 0) pixels are filled by preceding valid (mask value >0)
    // pixels.
    return dailyCol.mosaic() 
      .clip(aoi)
      .copyProperties(firstImage, ["system:time_start", "system:index"])
      .set("date_str", date);
  });

  return ee.ImageCollection.fromImages(mosaicedImages);
  // https://developers.google.com/earth-engine/apidocs/ee-imagecollection-fromimages
};

// ==============================================================================
// 4A. OLD PIPELINE
// ==============================================================================

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PERCENTAGE))

var baselineImageOld = prep(s2.filterDate(BASELINE_START, BASELINE_END))
  .median()
  .clip(aoi);

var monitoringImagesRawOld = prep(s2.filterDate(MONITORING_START, MONITORING_END));
var monitoringImagesOld = dailyMosaic(monitoringImagesRawOld).sort('system:time_start');

var imageListOld = monitoringImagesOld.toList(21)
var secondImage = ee.Image(imageListOld.get(1))
var thirdImage = ee.Image(imageListOld.get(2))

Map.addLayer(
  monitoringImagesOld.first(),
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.1, gamma: 1.4},
  'Old Cloud Masking: First monitoring acquisition', false
)

Map.addLayer(
  secondImage,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.1, gamma: 1.4},
  'Second monitoring acquisition', false
)

Map.addLayer(
  thirdImage,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.1, gamma: 1.4},
  'Third monitoring acquisition'
)

Map.addLayer(
  baselineImageOld,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.1, gamma: 1.4},
  'Baseline RGB'
  )

//==============================================================================
// 4B. NEW PIPELINE
// ==============================================================================

var maskedBaselineCollection = cloudMasking.build(baselineParams);
var maskedMonitoringCollection = cloudMasking.build(monitoringParams);

var baselineImage = maskedBaselineCollection
  .map(addNDVI)
  .select(BANDS.concat("NDVI"))
  .median()
  .clip(aoi);

var monitoringImagesRaw = maskedMonitoringCollection
  .map(addNDVI)
  .select(BANDS.concat("NDVI"))

var monitoringImages = dailyMosaic(monitoringImagesRaw).sort("system:time_start");

Map.addLayer(
  monitoringImages.first(),
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 600, gamma: 1},
  'New Cloud Masking : First monitoring acquisition'
)

stop
  
// Map.addLayer(
//   thirdImage.mask().select('B3'), 
//   {min: 0, max: 1, palette: ['red', 'green']}, 
//   'Internal Mask (Green=Valid, Red=Masked)'
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
  max: 21,
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
  max: 21, // can be made dynamic if so wished
  palette: [
    //'#ffffbf', // Yellow: Moderately occluded
    'black', // low cloud occurrence
    'white'  // high cloud occurrence
  ]
};

Map.addLayer(
  alerts.fromClassCollection.first(),
  fromClassParams,
  "First image of the fromClassCollection"
)

// Map.addLayer(
//   alerts.toClassCollection.first(),
//   toClassParams,
//   "First image of the toClassCollection"
// )

// Map.addLayer(
//   alerts.changeEvents.first().select("delta_ndvi"),
//   ndviParams,
//   "Delta NDVI of the first monitoring image"
// )

// Map.addLayer(
//   alerts.changeEvents.first().select("delta_ndvi_thresholded"),
//   ndviParams,
//   "Delta NDVI thresholded >=0.2 of the first monitoring image"
// )

Map.addLayer(
  alerts.changeReport.select("to_class_count"),
  imageCountVis,
  "L16 - To Class Count"
)

Map.addLayer(
  alerts.changeReport.select("from_class_count"),
  imageCountVis,
  "L15 - From Class Count"
)

Map.addLayer(
  alerts.changeReport.select('occluded_count'),
  occludivityVis,
  'L01 - Occluded Pixel Count', false
);

Map.addLayer(
  alerts.changeReport.select("available_image_count"),
  {palette: "red"}, "L0 - Available Image Count", false
)

Map.addLayer(
  classifiedMonitoringCollection.first().select("classification"),
  visClassParams,
  'First monitoring acquisition - CLASSIFIED', false
)

Map.addLayer(
  classifiedBaselineImage.select('classification'),
  visClassParams,
  'Baseline class map', true
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