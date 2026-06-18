var pyeo = require('users/matthewjpayne1/a4f:pyeoChangeAlerts')
var cloudMasking = require('users/matthewjpayne1/a4f:cloudMasking');
// ==============================================================================
// 1. PARAMETERS & CONSTANTS 
// ==============================================================================
 
var aoi = ee.Geometry.Rectangle([35.27456, -0.42817, 35.33481, -0.37977])

var BASELINE_START = '2020-01-01';
var BASELINE_END = '2020-12-31';
var MONITORING_START = '2024-06-30';
var MONITORING_END = '2024-12-31';

var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
//var MAX_CLOUD_PERCENTAGE = 10; // redundant as per image property, per pixel cloud probability is used
var MAX_CLOUD_PROBABILITY_PER_PIXEL = 30; // 100 = minimal discrimination
var MAX_CLOUD_SCORE_PER_PIXEL = 30; // 100 = no discrimination 

var FOREST = 1;
var SOIL = 2;
var AGRICULTURE = 3;
var URBAN = 4;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, AGRICULTURE];
var allClasses = [FOREST, SOIL, AGRICULTURE];

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
//Map.centerObject(aoi, 14)
// Map.addLayer(aoi, {color: 'red'}, 'AOI Outline', false);

// ==============================================================================
// 3. VISUALISATION PARAMETERS
// ==============================================================================

var classColourMap = {
  1: "green", // forest
  2: "brown", // soil
  3: "orange", // agriculture
  4: "grey" // urban
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
  max: 36,
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
  max: 17,
  palette: [
    "#ffffbf", // Yellow: few changes
    "#d7191c", // Red: max changes
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
  min: -1,
  max: 1,
  palette: ["white", "green"] // white low value, green high value
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
  min: 1720080614769,
  max: 1735200612936, // Milliseconds
  palette: ['#ffffb2', '#fecc5c', '#fd8d3c'] // pale yellow to orange
}

// visual parameters for min and max counts of post-fcd changes
// hardcoded, only works for the aoi, classifier and time range of this test
var postFCDChangeCountVisParams = {
  min: 1,
  max: 21,
  palette: ["#FA8FF1", "#700567"] // light pink, dark pink
};

var postFCDNoChangeCountVisParams = {
  min: 0,
  max: 17,
  palette: ["#8FCBFA", "#054270"] // light blue, dark blue
}

var postFCDOccludedCountVisParams = {
  min: 0,
  max: 28,
  palette: ["black", "white"]
}

var postFCDValidImageCountVisParams = {
  min: 1,
  max: 23,
  palette: ["#1E6D08", "#72F24E"] // dark green, light green
}

var postFCDChangeRepeatabilityVisParams = {
  min: 0,
  max: 100,
  palette: ["white", "red"]
}

var binaryTimeSeriesDecisionVisParams = {
  min: 0,
  max: 1,
  palette: ["green", "red"] // green no alert, red yes alert
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

// ==============================================================================
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

var imageList = monitoringImages.toList(36)
// var secondImage = ee.Image(imageList.get(1))
// var thirdImage = ee.Image(imageList.get(2))
var finalImage = ee.Image(imageList.get(35))


// Map.addLayer(
//   thirdImage.mask().select('B3'), 
//   {min: 0, max: 1, palette: ['red', 'green']}, 
//   'Internal Mask (Green=Valid, Red=Masked)'
// )

// 0 = masked and 1 = valid

// Inline training: forest / non-forest points within the AOI. Test fixture
// only — disappears once the SEPAL CLASSIFICATION recipe wrapper is in place.
var trainingPoints = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([35.32888, -0.40446]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.296289, -0.388055]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.308155, -0.397947]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.31852, -0.40181]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.33311, -0.40293]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.28316, -0.40649]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.29303, -0.41639]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.27982, -0.42553]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.32432, -0.42635]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.29419, -0.4224]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([35.31159, -0.41248]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.28713, -0.39244]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.31117, -0.38571]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.30584, -0.38249]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.3058, -0.38686]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.28021, -0.40916]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.316158, -0.404154]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.279283, -0.380488]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.324634, -0.404197]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.314549, -0.399527]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([35.31506, -0.40261]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.314141, -0.399462]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.320149, -0.399226]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.31751, -0.396437]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.316337, -0.392793]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.306815, -0.391821]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.295874, -0.410485]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.276691, -0.401709]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.291149, -0.382334]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.301513, -0.395688]), {'class': AGRICULTURE}), 
    ee.Feature(ee.Geometry.Point([35.286128, -0.393778]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.29911, -0.383282]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.31654, -0.37993]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.313554, -0.397816]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.312223, -0.382667]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.30873, -0.414131]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.303945, -0.421941]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.284351, -0.410921]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.285961, -0.424954]), {'class': URBAN}), 
    ee.Feature(ee.Geometry.Point([35.28382, -0.41076]), {'class': URBAN})
])

Map.addLayer(
  baselineImage,
  visParamsRGB,
  'Baseline Image', false
)

// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', FOREST)),
//     {color: '#ffa6eb'}, 'Training: forest')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', SOIL)),
//     {color: '#C49852'}, 'Training: non-forest')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', AGRICULTURE)),
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

Map.addLayer(
  classifiedBaselineImage.select('classification'),
  visClassParams,
  'Baseline class map', false
)

// ==============================================================================
// 6. RUN CHANGE DETECTION
// ==============================================================================

var alerts = pyeo.run_change_detection({
    aoi: aoi,
    classifiedBaseline: classifiedBaselineImage,
    classifiedMonitoringCollection: classifiedMonitoringCollection,
    changeFromClasses: changeFromClasses,
    changeToClasses: changeToClasses,
    minRequiredValidatedDetectionsThreshold: 2,
    minRequiredClassifierDetectionsThreshold: 5,
    percentageProbabilityThreshold: 50,
    minRequiredFromDetectionsThreshold: 2,
    minRequiredToDetectionsThreshold: 2,
    dNdviGate: {use_ndvi: false,
      band: 'NDVI',
      threshold: 0.1,  // -2.0 switches off delta ndvi threshold
      minRequiredDeltaNDVIDetectionsThreshold: 10
      }
});

// ==============================================================================
// 7. CHECKING THE CHANGE REPORT
// ==============================================================================

// // get min and max change dates from the test area, then hardcode earlier for vis
// var dateStats = alerts.changeReport.select("first_change_date_above_threshold").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("Change Date Min/Max (Milliseconds):", dateStats);

// // // get min and max change counts from the test area, then hardcode earlier for vis
// var changeStats = alerts.changeReport.select("post_fcd_change_count").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("min and max count of post-fcd changes", changeStats)

// // // get min and max no-change counts from the test area, then hardcode earlier for vis
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

// var unconfirmedChangeCountStats = alerts.changeReport.select("total_changes").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("min and max count of unconfirmed change stats", unconfirmedChangeCountStats)

// // get min and max change dates from the test area, then hardcode earlier for vis
// var dNDVIStats = alerts.changeReport.select("deltaNDVI_change_count").reduceRegion({
//     reducer: ee.Reducer.minMax(),
//     geometry: aoi,
//     scale: 10
// });
// print("count of delta ndvi:", dNDVIStats);

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

Map.addLayer(
  alerts.changeEvents.first().select("delta_ndvi"),
  visParamsNDVI,
  "Delta NDVI of the first monitoring image", false
)

Map.addLayer(
  alerts.changeEvents.first().select("delta_ndvi_thresholded_mask"),
  {},
  //visParamsNDVI,
  "Delta NDVI above threshold of the first monitoring image", false
)

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
  alerts.changeReport.select("binary_decision_from_to_map"),
  binaryTimeSeriesDecisionVisParams,
  "L17 - Binary Decision Thresholds on FROM and TO counts", false
)

Map.addLayer(
  alerts.changeReport.select("to_class_count"),
  imageCountVisParams,
  "L16 - To Class Count", false
)

Map.addLayer(
  alerts.changeReport.select("from_class_count"),
  imageCountVisParams,
  "L15 - From Class Count", false
)

Map.addLayer(
  alerts.changeReport.select("binary_combined_delta_decision_map"),
  binaryTimeSeriesDecisionVisParams,
  "L14 - Binary dNDVI & dClass Decision Map", false
)

Map.addLayer(
  alerts.changeReport.select("binary_delta_class_decision_map"),
  binaryTimeSeriesDecisionVisParams,
  "L13 - Binary dClass Decision Map", false
)

Map.addLayer(
  alerts.changeReport.select("binary_delta_ndvi_decision_map"),
  binaryTimeSeriesDecisionVisParams,
  "L12 - Binary dNDVI Decision Map", false
)

Map.addLayer(
  alerts.changeReport.select("deltaNDVI_change_count"),
  {min: 1,
  max: 23,
  palette: ["white", "green"]
  },
  "L11 - dNDVI only change detection count", false
)

Map.addLayer(
  alerts.changeReport.select("fcd_decision_map"),
  dateVisParams,
  "L10 - FCD Decision Map", false
)

Map.addLayer(
  alerts.changeReport.select("binary_timeseries_decision"),
  binaryTimeSeriesDecisionVisParams,
  "L09 - Binary timeseries decision", false
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_change_repeatability_pct"),
  postFCDChangeRepeatabilityVisParams,
  "L08 - Post-FCD Change Detection Repeatability", false
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_valid_image_count"),
  postFCDValidImageCountVisParams,
  "L07 - Post-FCD Valid Image Count", false
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_occluded_count"),
  postFCDOccludedCountVisParams,
  "L06 - Post-FCD Occluded Count", false
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_nochange_count"),
  postFCDNoChangeCountVisParams,
  "L05 - Post-FCD Combined Non-Alert Count", false
)

Map.addLayer(
  alerts.changeReport.select("post_fcd_change_count"),
  postFCDChangeCountVisParams,
  "L04 - Post-FCD Combined Alert Count", false
)

Map.addLayer(
  alerts.changeReport.select("first_change_date_above_threshold"),
  dateVisParams,
  "L03 - FCD & Combined Alert Detection", false
)

Map.addLayer(
  alerts.changeReport.select("total_changes"),
  changeDetectionCountVisParams,
  "L02 - Class Change Detection Count", false
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

Map.addLayer(
  finalImage,
  visParamsRGB,
  "Final Image of Monitoring Stack", false
)

Map.addLayer(
  baselineImage.select("NDVI"),
  visParamsNDVI,
  "NDVI Baseline", false
)

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

// ==============================================================================
// 8. VISUALISING CHANGE REPORT LAYERS COMPARISON
// ==============================================================================

// create separate map instances for a multi "panel" visualisation
var mapBaseline = ui.Map();
var mapFinalChangeImage = ui.Map();
//var mapChangeReport = ui.Map();

// label map instance titles
mapBaseline.add(ui.Label("Baseline Image", {position: "top-center"})); 
mapFinalChangeImage.add(ui.Label("Final Image of Change Period", {position: "top-center"}));
//mapChangeReport

// add layers to the map instances
mapBaseline.addLayer(
  baselineImage,
  visParamsRGB,
  "Baseline Image"
  );

mapFinalChangeImage.addLayer(
  finalImage,
  visParamsRGB,
  "Final Image of Monitoring Stack"
  )
  
// synchronise the maps together
var linker = ui.Map.Linker([mapBaseline, mapFinalChangeImage]);

// create a layout panel holding the maps side by side
var mapGrid = ui.Panel(
  [mapBaseline, mapFinalChangeImage],
  ui.Panel.Layout.Flow("horizontal"),
  {stretch: "both"}
)
// replace default map instance of the code editor with the new grid
ui.root.widgets().reset([mapGrid]);

// center the map
mapBaseline.centerObject(aoi, 14)
// MapBaseline.addLayer(aoi, {color: 'red'}, 'AOI Outline', false);