var pyeo = require('users/mp730/A4F:pyeoChangeAlerts')
var cloudMasking = require('users/mp730/A4F:cloudMasking');
var project_asset_path = 'projects/aim4forests-499914/assets/'
var aoi_friendly_name = 'Ghana'

// ==============================================================================
// 1. PARAMETERS & CONSTANTS 
// ==============================================================================

var inspection_marker = ee.Geometry.Point(-2.24103, 6.59954);

// construct a roughly 30 km2 square Area Of Interest
var centroid = [-2.24669, 6.58864] //-2.11734, 6.64627 //  -2.0571, 6.6856 //  -2.00664, 6.63018
var lon = centroid[0]
var lat = centroid[1]
var aoi = ee.Geometry.Rectangle([lon - 0.025, lat - 0.025, lon + 0.025, lat + 0.025]);
var CRS = "EPSG:32630";

//print("AOI area (km2)", aoi.area().divide(1000 * 1000))

var BASELINE_START = '2021-01-01';
var BASELINE_END = '2022-12-31';
var MONITORING_START = '2023-01-01';
var MONITORING_END = '2023-12-31';

// var BASELINE_START = '2021-01-01';
// var BASELINE_END = '2021-12-31';
// var MONITORING_START = '2022-01-01';
// var MONITORING_END = '2022-12-31';

var BANDS = ['B2', 'B3', 'B4', 'B6', 'B8', 'B11', 'B12'];
var MAX_CLOUD_PROBABILITY_PER_PIXEL = 20; // 100 = minimal discrimination
var MAX_CLOUD_SCORE_PER_PIXEL = 20; // 100 = no discrimination 

var FOREST = 1;
var SOIL = 2;
var GRASSLAND = 3;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, GRASSLAND];
var allClasses = [FOREST, SOIL, GRASSLAND];
var changeFromClassesStr = ["Forest"];
var changeToClassesStr = ["Soil", "Grassland"];
var allClassesStr = ["Forest", "Soil", "Grassland"];

// change detection parameters
var minRequiredValidatedDetectionsThreshold = 2;
var minRequiredClassifierDetectionsThreshold = 5;
var percentageProbabilityThreshold = 50;
var minRequiredFromDetectionsThreshold = 2;
var minRequiredToDetectionsThreshold = 2
var useNdvi = true
var deltaNdviThreshold = 0.2;  // -2.0 switches off delta ndvi threshold
var minRequiredDeltaNDVIDetectionsThreshold = 10;

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

// create a dictionary of pipeline parameters to export as metadata with the images as assets
// this is added to the images later
var pipelineParams = {
  'baseline_start': BASELINE_START,
  'baseline_end': BASELINE_END,
  'monitoring_start': MONITORING_START,
  'monitoring_end': MONITORING_END,
  'bands': JSON.stringify(BANDS),
  'max_cloud_probability_per_pixel': MAX_CLOUD_PROBABILITY_PER_PIXEL,
  'max_cloud_score_per_pixel': MAX_CLOUD_SCORE_PER_PIXEL,
  'change_from_classes': JSON.stringify(changeFromClasses),
  'change_from_classes_str': JSON.stringify(changeFromClassesStr),
  'change_to_classes': JSON.stringify(changeToClasses),
  'change_to_classes_str': JSON.stringify(changeToClassesStr),
  'all_classes': JSON.stringify(allClasses),
  'all_classes_str': JSON.stringify(allClassesStr),
  'min_validated_detections_threshold': minRequiredValidatedDetectionsThreshold,
  'min_classifier_detections_threshold': minRequiredClassifierDetectionsThreshold,
  'percentage_probability_threshold': percentageProbabilityThreshold,
  'min_from_detections_threshold': minRequiredFromDetectionsThreshold,
  'min_to_detections_threshold': minRequiredToDetectionsThreshold,
  'use_ndvi': useNdvi,
  'delta_ndvi_threshold': deltaNdviThreshold,
  'min_delta_ndvi_detections_threshold': minRequiredDeltaNDVIDetectionsThreshold
};

// ==============================================================================
// 2. MAP INITIALISATION
// ==============================================================================

Map.centerObject(aoi, 14)
Map.addLayer(aoi, {color: 'red'}, 'AOI Outline', false);


// ==============================================================================
// 3. VISUALISATION PARAMETERS
// ==============================================================================

// named CSS colours https://www.w3schools.com/cssref/css_colors.php
var classColourMap = {
  1: "ForestGreen", // forest
  2: "LightSalmon", // soil
  3: "LightGreen" // grassland
}

// total changes palette
var totalChangesPalette = ["#FFE2E2", "#9F0712"]; // reds

// fcd decision map palette
var fcdDecisionMapPalette = ["#DBEAFE", "#1C398E"]; // blues

// fcd repeatability palette
var fcdRepeatabilityPalette = ["#DCFCE7", "#0D542B"]; // greens

// RGB parameters
var visParamsRGB = {
  min: 0,
  max: 2500,
  gamma: 1.4,
  bands: ["B4", "B3", "B2"]
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
  bands: ["classification"],
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

// temporarily remove any problematic images
// var cleanedMonitoringCollection = monitoringImages
  // .filter(ee.Filter.neq("system:index", "20190107T102411_20190107T103313_T30NXN"))
  // .filter(ee.Filter.neq("system:index", "20190112T102409_20190112T103742_T30NXN"))
  // .filter(ee.Filter.neq("system:index", "20190117T102351_20190117T103132_T30NXN"))
  //.filter(ee.Filter.neq("system:index", "20190107T102411_20190107T103313_T30NWN"))
var listLength = monitoringImages.size();
var imageList = monitoringImages.toList(listLength);
var firstImage = ee.Image(imageList.get(8));
var finalImage = ee.Image(imageList.get(listLength.subtract(1))); // 8

Map.addLayer(
  baselineImage,
  visParamsRGB,
  'Baseline Image'
)


// Map.addLayer(
//   baselineImage,
//   {min: 0,
//   max: 5000,
//   gamma: 1.4,
//   bands: ["B8", "B4", "B3"] // R - NIR, G - Red, B - Green
// },
//   'Baseline Image'
// )


Map.addLayer(
  firstImage,
  visParamsRGB,
  "Beginning Monitoring Image")

Map.addLayer(
  finalImage,
  visParamsRGB,
  "Ending Monitoring Image")
  
var trainingPoints = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([-2.26487, 6.611966]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.257596, 6.605721]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.267102, 6.605828]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.268432, 6.610709]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.249335, 6.610645]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.247812, 6.60376]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.264355, 6.601863]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.267681, 6.601096]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.240901, 6.610874]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.233069, 6.610299]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.247639, 6.60367]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.245322, 6.607272]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.250021, 6.60546]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.226632, 6.607762]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.223285, 6.60384]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.233413, 6.602625]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.222147, 6.611642]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.23646, 6.602647]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.256267, 6.575481]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.24586, 6.579488]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.249961, 6.570721]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.268071, 6.568441]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.248673, 6.57151]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.234979, 6.570152]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.246824, 6.57352]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.249442, 6.574863]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.240516, 6.575076]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.230903, 6.572348]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.250579, 6.57158]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.252725, 6.574991]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.253605, 6.56802]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.248176, 6.565548]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.254721, 6.572689]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.23719, 6.564418]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.248433, 6.565185]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.252296, 6.57011]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.242275, 6.570792]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.233355, 6.573078]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.234814, 6.578194]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.222712, 6.567152]), {'class': FOREST}),

    ee.Feature(ee.Geometry.Point([-2.247789, 6.609233]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.244227, 6.605482]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.24294, 6.60465]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.240687, 6.601347]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.248068, 6.612111]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.22405, 6.59672]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.232264, 6.591735]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.225548, 6.591905]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.239452, 6.593611]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.241619, 6.593696]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.244409, 6.592673]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.246898, 6.5915]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.249173, 6.590349]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.225616, 6.586955]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.227461, 6.584845]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.268176, 6.584198]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.266867, 6.584837]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.268562, 6.58456]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.268348, 6.585029]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.268479, 6.565755]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.260175, 6.564305]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.245133, 6.572]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.242515, 6.565712]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.255561, 6.56503]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.245133, 6.572043]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.242704, 6.571794]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.242726, 6.571794]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.238842, 6.568212]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.234679, 6.566677]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.245558, 6.570344]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.248669, 6.569086]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.227631, 6.584617]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.22555, 6.586898]), {'class': SOIL}),
    
    ee.Feature(ee.Geometry.Point([-2.224586, 6.601175]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.223063, 6.598298]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.232933, 6.598873]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.227655, 6.602006]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.22553, 6.601218]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.242971, 6.597042]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.240826, 6.595017]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.243057, 6.599515]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.226238, 6.583054]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.225938, 6.58655]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.250068, 6.582922]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.249339, 6.588016]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.252364, 6.578381]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.251783, 6.583708]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.256568, 6.58618]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.256138, 6.588205]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.261782, 6.586393]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.253328, 6.58034]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.249701, 6.5878]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.261846, 6.586543]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.27103, 6.590955]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.263262, 6.578806]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.264164, 6.576376]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.265237, 6.584455]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-2.240473, 6.571197]), {'class': GRASSLAND})
    // ee.Feature(ee.Geometry.Point([]), {'class': GRASSLAND}),
    // ee.Feature(ee.Geometry.Point([]), {'class': GRASSLAND}),
    // ee.Feature(ee.Geometry.Point([]), {'class': GRASSLAND}),
    // ee.Feature(ee.Geometry.Point([]), {'class': GRASSLAND}),
    // ee.Feature(ee.Geometry.Point([]), {'class': GRASSLAND})
])

// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', FOREST)),
//     {color: 'ForestGreen'}, 'Training: Forest')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', SOIL)),
//     {color: 'LightSalmon'}, 'Training: Soil')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', GRASSLAND)),
//     {color: 'LightGreen'}, 'Training: Grassland')

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

// // Pre-classify to match the contract runPyeoChangeAlerts expects.
var classifiedBaselineImage = baselineImage
    .classify(classifier).rename('classification')
    .addBands(baselineImage.select('NDVI'))
    .clip(aoi)

Map.addLayer(
  classifiedBaselineImage.select('classification'),
  visClassParams,
  'Baseline class map', false
)

var classifiedMonitoringCollection = monitoringImages.map(function (img) {
    return img.classify(classifier).rename('classification')
        .addBands(img.select('NDVI'))
        .copyProperties(img, ['system:time_start'])
})

// ==============================================================================
// 6. RUN CHANGE DETECTION
// ==============================================================================


var alerts = pyeo.run_change_detection({
    aoi: aoi,
    classifiedBaseline: classifiedBaselineImage,
    classifiedMonitoringCollection: classifiedMonitoringCollection,
    changeFromClasses: changeFromClasses,
    changeToClasses: changeToClasses,
    minRequiredValidatedDetectionsThreshold: minRequiredValidatedDetectionsThreshold,
    minRequiredClassifierDetectionsThreshold: minRequiredClassifierDetectionsThreshold,
    percentageProbabilityThreshold: percentageProbabilityThreshold,
    minRequiredFromDetectionsThreshold: minRequiredFromDetectionsThreshold,
    minRequiredToDetectionsThreshold: minRequiredToDetectionsThreshold,
    dNdviGate: {use_ndvi: useNdvi,
      band: 'NDVI',
      threshold: deltaNdviThreshold,
      minRequiredDeltaNDVIDetectionsThreshold: minRequiredDeltaNDVIDetectionsThreshold
      }
});


// ==============================================================================
// 7. CHECKING THE CHANGE REPORT
// ==============================================================================

Map.addLayer(inspection_marker, {color: "pink", size: 14}, "Inspection Marker")

// create a client-side object of the point inspector, so the date string can be formatted
//    into a readable human date
var changeReportAtPoint = alerts.changeReport.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: inspection_marker,
  scale: 10
})

print("pixel properties at the inspection marker:", changeReportAtPoint)

changeReportAtPoint.evaluate(function(result) {
  if (result.first_change_date_above_threshold > 0) {
    // use JS to create a Date object, which has .toDateString()
    // Date is an EE function that returns a string, strings don't have .toDateString()
    var readableFirstChange = new Date(result.first_change_date_above_threshold).toDateString();

    print("First change was on: ", readableFirstChange);
  }
  else {
    print("No change at this location")
  }
})

// ==============================================================================
// 8. SAVING CHANGE REPORT AND PARAMETERS AS ASSETS FOR FIGURE CREATION (PYTHON)
// ==============================================================================

// get minMax date stats for dates of change and total changes colour ramps
var combinedStats = alerts.changeReport
  .select(["fcd_decision_map", "total_changes", "first_change_date_above_threshold"])
  .reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: aoi,
    scale: 10,
    maxPixels: 1e9
  });

// send for evaluation to get client-side numbers for exporting
combinedStats.evaluate(function(stats) {
  
  var fcdDecisionVisParams = {
    bands: ["fcd_decision_map"],
    min: stats.fcd_decision_map_min,
    max: stats.fcd_decision_map_max,
    palette: fcdDecisionMapPalette
  };
  
  var totalChangesVisParams = {
    bands: ["total_changes"],
    min: stats.total_changes_min,
    max: stats.total_changes_max,
    palette: totalChangesPalette
  };
  
  var fcdVisParams = {
    bands: ["first_change_date_above_threshold"],
    min: stats.first_change_date_above_threshold_min,
    max: stats.first_change_date_above_threshold_max,
    palette: fcdDecisionMapPalette
  };
  
  var repeatabilityVisParams = {
    bands: ["post_fcd_change_repeatability_pct"],
    min: 0,
    max: 100,
    palette: fcdRepeatabilityPalette
  };
  
  Map.addLayer(
  alerts.changeReport.select("total_changes"),
  totalChangesVisParams,
  "L2 - Total Changes");
  
  Map.addLayer(
  alerts.changeReport.select("first_change_date_above_threshold"),
  fcdVisParams,
  "L3 - FCD Map");
  
  Map.addLayer(
  alerts.changeReport.select("post_fcd_change_repeatability_pct"),
  repeatabilityVisParams,
  "L8 - Post-FCD Change Repeatability");

  Map.addLayer(
  alerts.changeReport.select("fcd_decision_map"),
  fcdDecisionVisParams,
  "L10 - FCD Decision Map");
  
//   var changeReportWithMetadata = alerts.changeReport
//     .set(pipelineParams)
//     .set("fcdDecisionVisParams", JSON.stringify(fcdDecisionVisParams))
//     .set("totalChangesVisParams", JSON.stringify(totalChangesVisParams))
//     .set("repeatabilityVisParams", JSON.stringify(repeatabilityVisParams));

//   Export.image.toAsset({
//     image: changeReportWithMetadata,
//     description: aoi_friendly_name + "_change_report",
//     assetId: project_asset_path + aoi_friendly_name + "_" + "change_report",
//     region: aoi,
//     scale: 10,
//     crs: CRS,
//     maxPixels: 1e13
//   });

//   // Export.image.toDrive({
//   //   image: changeReportWithMetadata.toDouble(),
//   //   description: aoi_friendly_name + "_change_report_Drive",
//   //   fileNamePrefix: aoi_friendly_name + "_" + "change_report",
//   //   region: aoi,
//   //   scale: 10,
//   //   crs: CRS,
//   //   maxPixels: 1e13,
//   //   fileFormat: "GeoTIFF"
//   // });

});

// /// assign metadata to the assets
// var baselineImageWithMetadata = baselineImage.set(pipelineParams).set("visParamsRGB", JSON.stringify(visParamsRGB));
// var firstImageWithMetadata = firstImage
//   .set(pipelineParams)
//   .set("visParamsRGB", JSON.stringify(visParamsRGB))
//   .set("date", firstImage.get("date_str"));
// var finalImageWithMetadata = finalImage
//   .set(pipelineParams)
//   .set("visParamsRGB", JSON.stringify(visParamsRGB))
//   .set("date", finalImage.get("date_str"));
// var baseline_filename = aoi_friendly_name + "_baseline_" + BASELINE_START + "_" + BASELINE_END
// var firstImage_filename = aoi_friendly_name +  "_first_monitoring_" + MONITORING_START + "_" + MONITORING_END
// var finalImage_filename = aoi_friendly_name + "_final_monitoring_" + MONITORING_START + "_" + MONITORING_END

// Export.image.toAsset({
//   image: baselineImageWithMetadata,
//   description: baseline_filename,
//   assetId: project_asset_path + baseline_filename,
//   region: aoi,
//   scale: 10,
//   crs: CRS,
//   maxPixels: 1e13
// });

// Export.image.toAsset({
//   image: ee.Image(firstImageWithMetadata),
//   description: firstImage_filename,
//   assetId: project_asset_path + firstImage_filename,
//   region: aoi,
//   scale: 10,
//   crs: CRS,
//   maxPixels: 1e13
// });

// Export.image.toAsset({
//   image: finalImageWithMetadata,
//   description: finalImage_filename,
//   assetId: project_asset_path + finalImage_filename,
//   region: aoi,
//   scale: 10,
//   crs: CRS,
//   maxPixels: 1e13
// });

// Export.image.toDrive({
//   image: baselineImageWithMetadata.toDouble(),
//   description: baseline_filename,
//   fileNamePrefix: baseline_filename,
//   region: aoi,
//   scale: 10,
//   crs: CRS,
//   maxPixels: 1e13,
//   fileFormat: "GeoTIFF"
// });

// Export.image.toDrive({
//   image: classifiedBaselineImage
//     .select("classification")
//     .visualize(visClassParams),
//   description: baseline_filename + "_classified_Drive",
//   fileNamePrefix: baseline_filename + "_classified",
//   region: aoi,
//   scale: 10,
//   maxPixels: 1e13,
//   fileFormat: "GeoTIFF"
// });

// ==============================================================================
// 9. EXPORTING MONITORING COLLECTION TO DRIVE FOR MANUAL INSPECTION OF CHANGE
// ==============================================================================

var exportCollectionToDrive = function(collection, region, visParams, taskString) {
  // 1. Prepare the collection for export
  var toExport = collection.map(function(img) {
    var dateStr = img.date().format("YYYY-MM-dd");
    var visual = img.visualize(visParams);
    return visual.set("date_str", dateStr);
  });
  
  // 2. Convert collection to an ee.List to allow indexing
  var size = toExport.size();
  var collectionList = toExport.toList(size);
  
  // 3. Get an ee.List of all the date strings
  var datesList = toExport.aggregate_array("date_str");
  
  // 4. Evaluate the dates list to bring it to the client side
  datesList.evaluate(function(dates, error) {
    if (error) {
      print("Error evaluating dates:", error);
      return;
    }
    
    // Now 'dates' is a standard JavaScript array, so a for-loop works perfectly!
    for (var i = 0; i < dates.length; i++) {
      var dateString = dates[i];
      var safeDate = dateString.replace(/-/g, "_");
      var taskName = "Export_Quicklook_" + safeDate;
      
      // Fetch the specific image from the server-side list using the index
      var img = ee.Image(collectionList.get(i));
      
      //Create the export task
      Export.image.toDrive({
        image: img,
        description: taskName,
        fileNamePrefix: taskString + safeDate,
        region: region, 
        scale: 10,
        maxPixels: 1e13,
        fileFormat: "GeoTIFF"
      });
    }
  });
};



// var taskString = "S2_Quicklook_" 
// exportCollectionToDrive(monitoringImages, aoi, visParamsRGB, taskString);

// var taskString = "S2_Quicklook_Classified_" 
// exportCollectionToDrive(classifiedMonitoringCollection, aoi, visClassParams, taskString);