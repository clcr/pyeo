var pyeo = require('users/mp730/A4F:pyeoChangeAlerts')
var cloudMasking = require('users/mp730/A4F:cloudMasking');
// ==============================================================================
// 1. PARAMETERS & CONSTANTS
// ==============================================================================
 
// Near Achiase, Ghana -0.9083, 5.8089 deforestation reported by GFW 2024 -> 2025

var corner_coordinate = [-0.91729, 5.78453]
var lon = corner_coordinate[0]
var lat = corner_coordinate[1]

var aoi = ee.Geometry.Rectangle([lon, lat, lon + 0.05, lat + 0.05]);

print("AOI Area (km2)", aoi.area().divide(1000 * 1000))

var BASELINE_START = '2020-01-01';
var BASELINE_END = '2021-12-31';
var MONITORING_START = '2024-06-30';
var MONITORING_END = '2025-06-30';

var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
//var MAX_CLOUD_PERCENTAGE = 10; // redundant as per image property, per pixel cloud probability is used
var MAX_CLOUD_PROBABILITY_PER_PIXEL = 10; // 100 = minimal discrimination
var MAX_CLOUD_SCORE_PER_PIXEL = 10; // 100 = no discrimination 

var FOREST = 1;
var SOIL = 2;
var GRASSLAND = 3;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, GRASSLAND];
var allClasses = [FOREST, SOIL, GRASSLAND];

// parameter objects for imagery acquisition and cloud masking
var baselineParams = {
  aoi: aoi,
  startDate: BASELINE_START,
  endDate: BASELINE_END,
  useSR: true,
  maxCloudProbability: MAX_CLOUD_PROBABILITY_PER_PIXEL,
  maxCsProbability: MAX_CLOUD_SCORE_PER_PIXEL,
  method: "BOTH"
}

var monitoringParams = {
  aoi: aoi,
  startDate:  MONITORING_START,
  endDate: MONITORING_END,
  useSR: true,
  maxCloudProbability: MAX_CLOUD_PROBABILITY_PER_PIXEL,
  maxCsProbability: MAX_CLOUD_SCORE_PER_PIXEL,
  method: "BOTH"
}

// ==============================================================================
// 2. MAP INITIALISATION
// ==============================================================================
Map.centerObject(aoi, 14)
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

var imageCountVisParams = {
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

var changeDetectionCountVisParams = {
  min: 0,
  max: 5,
  palette: [
    '#d7191c', // Red: Very few changes
    '#fdae61', // Orange
    '#ffffbf', // Yellow: max changes
    ]
};

var occludivityVisParams = {
  min: 22,
  max: 34, // can be made dynamic if so wished
  palette: [
    'black', // low cloud occurrence
    'white'  // high cloud occurrence
  ]
};

var visParamsNDVI = {
  min: -0.2,
  max: 1,
  palette: ["white", "green"] // specifies the upper and lower range
}

var visParamsRGB = {
  min: 0,
  max: 2500,
  gamma: 1.4,
  bands: ["B4", "B3", "B2"]
}

// first and last change date visual parameters
// hardcoded, only works for the aoi, classifier and time range of this test
var dateVisParams = {
  min: 1719225569038,
  max: 1704105562859, // Milliseconds
  palette: ['#ffffb2', '#fecc5c', '#fd8d3c'] // pale yellow to orange
}

// visual parameters for min and max counts of post-fcd changes
// hardcoded, only works for the aoi, classifier and time range of this test
var postFCDChangeCountVisParams = {
  min: 1,
  max: 15,
  palette: ["#FA8FF1", "#700567"] // light pink, dark pink
};

var postFCDNoChangeCountVisParams = {
  min: 0,
  max: 11,
  palette: ["#8FCBFA", "#054270"] // light blue, dark blue
}

var postFCDOccludedCountVisParams = {
  min: 0,
  max: 33,
  palette: ["black", "white"]
}

var postFCDValidImageCountVisParams = {
  min: 1,
  max: 13,
  palette: ["#1E6D08", "#72F24E"] // dark green, light green
}

// ==============================================================================
// 4. HELPER FUNCTIONS
// ==============================================================================

// NDVI calculation
var addNDVI = function (img) {
    return img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'));
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

//==============================================================================
// 5. PIPELINE
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
  
var monitoringImages = dailyMosaic(monitoringImagesRaw)

var imageList = monitoringImages.toList(40)
var secondImage = ee.Image(imageList.get(1))
var thirdImage = ee.Image(imageList.get(2))

// Map.addLayer(
//   thirdImage.mask().select('B3'), 
//   {min: 0, max: 1, palette: ['red', 'green']}, 
//   'Internal Mask (Green=Valid, Red=Masked)'
// )

// 0 = masked and 1 = valid

// Inline training: forest / non-forest points within the AOI. Test fixture
// only — disappears once the SEPAL CLASSIFICATION recipe wrapper is in place.
var trainingPoints = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([-0.905047, 5.824373]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-0.894597, 5.825163]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-0.888352, 5.824779]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-0.909681, 5.814494]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-0.896914, 5.81078]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-0.890026, 5.809264]), {'class': FOREST}),  
    ee.Feature(ee.Geometry.Point([-0.886399, 5.810688]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-0.904767, 5.802277]), {'class': FOREST}),  
    ee.Feature(ee.Geometry.Point([-0.914079, 5.801103]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-0.905362, 5.79244]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-0.913345, 5.824589]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.904355, 5.818847]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.88708, 5.824837]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.88487, 5.823044]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.880707, 5.818881]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.880376, 5.810586]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.879946, 5.807576]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.866713, 5.806738]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.872489, 5.788479]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.867168, 5.788799]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-0.88658, 5.812399]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.898434, 5.81413]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.90249, 5.815825]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.91238, 5.816756]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.903743, 5.824949]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.910352, 5.826561]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.89854, 5.824276]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.912551, 5.820658]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.915854, 5.818673]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-0.906151, 5.809517]), {'class': GRASSLAND})
])

Map.addLayer(
  baselineImage,
  visParamsRGB,
  'Baseline Image', false
)

// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', FOREST)),
//     {color: '#0D5E39'}, 'Training: forest')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', SOIL)),
//     {color: '#C49852'}, 'Training: non-forest')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', GRASSLAND)),
//     {color: '#27F584'}, 'Training: non-forest')

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
    dNdviGate: {use_ndvi: false,  band: 'NDVI', threshold: 0.3} // -2.0 switches off delta ndvi threshold
});

// // get min and max change dates from the test area, then hardcode earlier for vis
// var dateStats = alerts.changeReport.select("first_change_date_above_threshold").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("Change Date Min/Max (Milliseconds):", dateStats);

// // get min and max change counts from the test area, then hardcode earlier for vis
// var changeStats = alerts.changeReport.select("post_fcd_change_count").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("min and max count of post-fcd changes", changeStats)

// // get min and max no-change counts from the test area, then hardcode earlier for vis
// var noChangeStats = alerts.changeReport.select("post_fcd_nochange_count").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("min and max count of post-fcd no-changes", noChangeStats)

// var postFCDOccludedStats = alerts.changeReport.select("post_fcd_occluded_count").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("min and max count of post-fcd occluded", postFCDOccludedStats)


// var postFCDValidImageCountStats = alerts.changeReport.select("post_fcd_valid_image_count").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("min and max count of post-fcd valid image counts", postFCDValidImageCountStats)

// Map.addLayer(
//   alerts.fromClassCollection.first(),
//   fromClassParams,
//   "First image of the fromClassCollection", false
// )

// Map.addLayer(
//   alerts.toClassCollection.first(),
//   toClassParams,
//   "First image of the toClassCollection"
// )

// Map.addLayer(
//   alerts.changeEvents.first().select("delta_ndvi"),
//   visParamsNDVI,
//   "Delta NDVI of the first monitoring image"
// )

// Map.addLayer(
//   alerts.changeEvents.first().select("delta_ndvi_thresholded"),
//   visParamsNDVI,
//   "Delta NDVI thresholded >=0.2 of the first monitoring image"
// )

Map.addLayer(
  monitoringImages.first(),
  visParamsRGB,
  'First monitoring acquisition', false
);

// Map.addLayer(
//   secondImage,
//   visParamsRGB,
//   'Second monitoring acquisition', false
// )

// Map.addLayer(
//   thirdImage,
//   visParamsRGB,
//   'Third monitoring acquisition', false
// )

Map.addLayer(
  alerts.changeReport.select("to_class_count"),
  imageCountVisParams,
  "L16 - To Class Count"
)

Map.addLayer(
  alerts.changeReport.select("from_class_count"),
  imageCountVisParams,
  "L15 - From Class Count"
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_valid_image_count"),
  postFCDValidImageCountVisParams,
  "L07 - Post-FCD Valid Image Count"
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_occluded_count"),
  postFCDOccludedCountVisParams,
  "L06 - Post-FCD Occluded Count", false
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_nochange_count"),
  postFCDNoChangeCountVisParams,
  "L05 - Post-FCD Combined Non-Alert Count"
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_change_count"),
  postFCDChangeCountVisParams,
  "L04 - Post-FCD Combined Alert Count"
)

Map.addLayer(
  alerts.changeReport.select("first_change_date_above_threshold"),
  dateVisParams,
  "L03 - FCD & Combined Alert Detection"
)

Map.addLayer(
  alerts.changeReport.select("total_changes"),
  changeDetectionCountVisParams,
  "L02 - Class Change Detection Count"
)

Map.addLayer(
  alerts.changeReport.select('occluded_count'),
  occludivityVisParams,
  'L01 - Occluded Pixel Count', false
);

Map.addLayer(
  alerts.changeReport.select("available_image_count"),
  {palette: "red"},
  "L00 - Available Image Count", false
)


// Map.addLayer(
//   classifiedMonitoringCollection.first().select("classification"),
//   visClassParams,
//   'First monitoring acquisition - CLASSIFIED', false
// )

// Map.addLayer(
//   classifiedBaselineImage.select('classification'),
//   visClassParams,
//   'Baseline class map', true
// )

// create a client-side object of the point inspector, so the date string can be formatted
//    into a readable human date
var changeReportAtPoint = alerts.changeReport.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: inspection_marker,
  scale: 10
})

print("pixel properties at the inspection marker:", changeReportAtPoint)

// changeReportAtPoint.evaluate(function(result) {
//   if (result.first_change_date_above_threshold > 0) {
//     // use JS to create a Date object, which has .toUTCString()
//     // Date is an EE function that returns a string, strings don't have .toUTCString()
//     var readableFirstChange = new Date(result.first_change_date_above_threshold).toUTCString();

//     print("First change was on: ", readableFirstChange);
//   }
//   else {
//     print("No change at this location")
//   }
// })