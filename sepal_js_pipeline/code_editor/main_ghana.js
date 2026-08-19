var pyeo = require('users/mp730/A4F:pyeoChangeAlertsSEPAL')
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

var BASELINE_START = '2021-01-01';
var BASELINE_END = '2022-12-31';
var MONITORING_START = '2023-01-01';
var MONITORING_END = '2023-12-31';

var BANDS = ['B2', 'B3', 'B4', 'B6', 'B8', 'B11', 'B12'];
var MAX_CLOUD_PROBABILITY_PER_PIXEL = 20; // 100 = minimal discrimination
var MAX_CLOUD_SCORE_PER_PIXEL = 20; // 100 = no discrimination 

var FOREST = 0;
var SOIL = 1;
var DISTURBED = 2;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, DISTURBED];
var allClasses = [FOREST, SOIL, DISTURBED];
var changeFromClassesStr = ["Forest"];
var changeToClassesStr = ["Soil", "Disturbed"];
var allClassesStr = ["Forest", "Soil", "Disturbed"];

// change detection parameters
var minRequiredValidatedDetectionsThreshold = 2;
var minRequiredClassifierDetectionsThreshold = 5;
var percentageProbabilityThreshold = 50;
var minRequiredFromDetectionsThreshold = 2;
var minRequiredToDetectionsThreshold = 2
var useIndex = true
var index = "NDVI"
var deltaIndexThreshold = 0.2;
var minRequiredDeltaIndexDetectionsThreshold = 2;
var useHazeFilter = false;
var fromAvailabilityThresholdPct = 0.1;
var hazeLikelihoodThresholdPct = 0.3;
var researchModeOn = true // whether to export imagery as quicklook geotiffs at 10 m spatial resolution, 
// but compressed to 0-255 bit range
// all export toggles below require research mode to be true. 
var exportChangeReport = false
var exportBaseline = false
var exportTimeSeriesRGBQuicklooks = false
var exportTimeSeriesClfQuicklooks = false
var exportClassifierPerformance = false

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
  'use_index': useIndex,
  'delta_index_threshold': deltaIndexThreshold,
  'min_delta_index_detections_threshold': minRequiredDeltaIndexDetectionsThreshold,
  "use_haze_filter": useHazeFilter,
  "haze_filter_from_availability_threshold_pct": fromAvailabilityThresholdPct,
  "haze_filter_likelihood_threshold_pct": hazeLikelihoodThresholdPct,
  "research_mode_on": researchModeOn
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
  0: "ForestGreen", // forest
  1: "LightSalmon", // soil
  2: "LightGreen" // disturbed vegetation
}

var classNameMap = {
  0: allClassesStr[0],
  1: allClassesStr[1],
  2: allClassesStr[2]
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
    return img.addBands(img.normalizedDifference(['B8', 'B4']).rename('gate_index'));
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
  .select(BANDS.concat("gate_index"))
  .median()
  .clip(aoi);

var monitoringImagesRaw = maskedMonitoringCollection
  .map(addNDVI)
  .select(BANDS.concat("gate_index"))
  
var monitoringImages = dailyMosaic(monitoringImagesRaw)

Map.addLayer(
  baselineImage,
  visParamsRGB,
  'Baseline Image', false
)

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
    ee.Feature(ee.Geometry.Point([-2.24101, 6.58565]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.23667, 6.57985]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.25899, 6.59221]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.25023, 6.59196]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.26366, 6.59622]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.26706, 6.58863]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.23083, 6.59226]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.269492, 6.576851]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.269321, 6.583203]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-2.265458, 6.57084]), {'class': FOREST}),
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
    ee.Feature(ee.Geometry.Point([-2.267256, 6.586498]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.265743, 6.588715]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.26644, 6.584985]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.265707, 6.590776]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.26679, 6.592119]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.260911, 6.588911]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.269998, 6.591991]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.265363, 6.591363]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.271189, 6.592194]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.259559, 6.58922]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.255174, 6.590734]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.253329, 6.590403]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.258833, 6.589497]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.260785, 6.588911]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.262459, 6.590286]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.2435, 6.567635]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.24689, 6.57192]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.244423, 6.572048]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.250259, 6.567401]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.2435, 6.567721]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-2.25728, 6.571046]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.263954, 6.573114]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.247388, 6.568232]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.259169, 6.567529]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.258911, 6.573839]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.265477, 6.566804]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.25874, 6.564097]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.265434, 6.566655]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.247496, 6.568232]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.2661, 6.573604]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.270437, 6.57742]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.25885, 6.573775]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.261189, 6.572453]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.269708, 6.573114]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.2657, 6.582524]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.251602, 6.580116]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.253233, 6.582141]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.270721, 6.577835]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.25171, 6.582724]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.248792, 6.582873]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.258558, 6.575582]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.265531, 6.577735]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.270038, 6.585153]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.269372, 6.580613]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.249588, 6.587093]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.254116, 6.586837]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.260339, 6.587775]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.261862, 6.586709]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.254223, 6.589629]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.261884, 6.584876]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.270885, 6.591015]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.270176, 6.58754]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.270091, 6.585537]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.252667, 6.588436]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.270198, 6.585622]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.265928, 6.583831]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.257976, 6.589267]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.230013, 6.592734]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.229949, 6.594546]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.224896, 6.588833]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.228457, 6.60088]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.225539, 6.601242]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.232277, 6.597874]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.230453, 6.595572]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.233135, 6.599537]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.227256, 6.598514]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.22996, 6.600453]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.226264, 6.583112]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.224398, 6.589166]), {'class': DISTURBED}),
    ee.Feature(ee.Geometry.Point([-2.243826, 6.573011]), {'class': DISTURBED})
])

if (researchModeOn) {
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', FOREST)),
    {color: classColourMap[0]}, 'Training: Forest')
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', SOIL)),
    {color: classColourMap[1]}, 'Training: Soil')
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', DISTURBED)),
    {color: classColourMap[2]}, 'Training: Disturbed')
}

var allSamples = baselineImage.sampleRegions({
    collection: trainingPoints,
    properties: ['class'],
    scale: 10
}).randomColumn("random")// add a column of 0 to 1, to use for training/validation split

var trainingSamples = allSamples.filter(ee.Filter.lt("random", 0.6))
var testingSamples = allSamples.filter(ee.Filter.gte("random", 0.6))

var classifier = ee.Classifier.smileRandomForest(50).train({
    features: trainingSamples,
    classProperty: 'class',
    inputProperties: BANDS.concat(['gate_index'])
})

// classify on testing dataset to assess how the model generalises to unseen data
var tested = testingSamples.classify(classifier);
var testedMatrix = tested.errorMatrix("class", "classification")

var trained = trainingSamples.classify(classifier)
var trainedMatrix = trained.errorMatrix("class", "classification")
print("Accuracy (%) on Trained Matrix:", trainedMatrix.accuracy())

print("full classification points size:", allSamples.size())

print("training points size:", trainingSamples.size())
print("testing points size:", testingSamples.size())

// Pre-classify to match the contract runPyeoChangeAlerts expects.
var classifiedBaselineImage = baselineImage
    .classify(classifier).rename('classification')
    .addBands(baselineImage.select('gate_index'))
    .clip(aoi)

var classifiedMonitoringCollection = monitoringImages.map(function (img) {
    return img.classify(classifier).rename('classification')
        .addBands(img.select('gate_index'))
        .copyProperties(img, ['system:time_start'])
})

Map.addLayer(
  classifiedBaselineImage.select("classification"),
  visClassParams,
  'Baseline class map'
)

// evaluate classifier performance
if (researchModeOn) {
  
  if (exportClassifierPerformance) {
    
    // make a dynamic dictionary
    // map over allClasses returning a list of integers as strings
    var allClassesKeys = allClasses.map(function(class_name) {
      return String(class_name)
    })
    var classDict = ee.Dictionary.fromLists(allClassesKeys, allClassesStr)
    
    var matrixArray = testedMatrix.array();
    var classIds = testedMatrix.order();
    
    // create list of column names for the predicted classes
    var colNames = classIds.map(function(id) {
      var idStr = ee.Number(id).format("%d");
      return classDict.get(idStr)
    })
    
    // convert the confusion matrix to a list of features
    var matrixList = matrixArray.toList();
    var featureList = matrixList.zip(classIds).map(function(rowWithId) {
      var rowWithIdList = ee.List(rowWithId);
      var rowValues = ee.List(rowWithIdList.get(0)); // count for the row
      var classId = ee.List(rowWithIdList.get(1)); // class ID
      var idStr = ee.Number(classId).format("%d");
      
      var className = classDict.get(idStr);      
      var featureProperties = ee.Dictionary.fromLists(colNames, rowValues);
      
      // update featureProperties
      featureProperties = featureProperties.set("Class", className)
      
      return ee.Feature(null, featureProperties)
    });
    
    // calculate testing accuracy and append to the list of features
    var testedAccuracy = testedMatrix.accuracy().format("%.2f");
    var firstColName = ee.String(colNames.get(0));
    var secondColName = ee.String(colNames.get(1));
    var accuracyProperties = ee.Dictionary({
      "Class": "Testing Dataset Accuracy"
      })
      .set(firstColName, testedAccuracy);
    var accuracyFeature = ee.Feature(null, accuracyProperties);
    
    var matrixFeatureCol = ee.FeatureCollection(featureList.add(accuracyFeature))
    var exportColumns = ["Class", allClassesStr[0], allClassesStr[1], allClassesStr[2], allClassesStr[3]]
    
    Export.table.toDrive({
      collection: matrixFeatureCol,
      description: 'Testing_Samples_Confusion_Matrix_Task',
      fileNamePrefix: 'Testing_Samples_Confusion_Matrix',
      fileFormat: 'CSV',
      selectors: exportColumns
    })
    
  }
  
}

// ==============================================================================
// 6. RUN CHANGE DETECTION
// ==============================================================================

var alerts = pyeo.runPyeoChangeAlerts({
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
    indexGate: {
      use: useIndex,
      index: index,
      threshold: deltaIndexThreshold,
      minRequiredDeltaIndexDetectionsThreshold: minRequiredDeltaIndexDetectionsThreshold
      },
    hazeFilter: {
      use: useHazeFilter,
      fromAvailabilityThresholdPct: fromAvailabilityThresholdPct,
      hazeLikelihoodThresholdPct: hazeLikelihoodThresholdPct
    }
});

// ==============================================================================
// 7. CHECKING THE CHANGE REPORT
// ==============================================================================

Map.addLayer(inspection_marker, {color: "pink", size: 14}, "Inspection Marker")

// create a client-side object of the point inspector, so the date string can be formatted
//    into a readable human date
var changeReportAtPoint = alerts.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: inspection_marker,
  scale: 10
})

print("pixel properties at the inspection marker:", changeReportAtPoint)

changeReportAtPoint.evaluate(function(result) {
  if (result.first_change_date_above_threshold > 0) {
    
    var fractionalYear = result.first_change_date_above_threshold;
    
    // get the whole year and the fraction
    var year = Math.floor(fractionalYear);
    var fraction = fractionalYear - year;
    
    // use Date.UTC to prevent the browser's local timezone from shifting the results
    var startOfYearMillis = new Date(Date.UTC(year, 0, 1)).getTime();
    var startOfNextYearMillis = new Date(Date.UTC(year + 1, 0, 1)).getTime();
    
    // calculate total milliseconds
    var millisInYear = startOfNextYearMillis - startOfYearMillis;
    
    // multiply the fraction by the year's total milliseconds and add to the start of the year
    var targetMillis = startOfYearMillis + (fraction * millisInYear);
    
    // convert back to a readable string
    var readableFirstChange = new Date(targetMillis).toDateString();

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
var combinedStats = alerts
  .select(["fcd_decision_map", "total_changes"])
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
  
  var repeatabilityVisParams = {
    bands: ["post_fcd_change_repeatability_pct"],
    min: 0,
    max: 100,
    palette: fcdRepeatabilityPalette
  };
  
  Map.addLayer(
  alerts.select("total_changes"),
  totalChangesVisParams,
  "L2 - Total Changes");
  
  Map.addLayer(
  alerts.select("post_fcd_change_repeatability_pct"),
  repeatabilityVisParams,
  "L8 - Post-FCD Change Repeatability");

  Map.addLayer(
  alerts.select("fcd_decision_map"),
  fcdDecisionVisParams,
  "L10 - FCD Decision Map");
  
  var changeReportWithMetadata = alerts
    .set(pipelineParams)
    .set("fcdDecisionVisParams", JSON.stringify(fcdDecisionVisParams))
    .set("totalChangesVisParams", JSON.stringify(totalChangesVisParams))
    .set("repeatabilityVisParams", JSON.stringify(repeatabilityVisParams));

  if (researchModeOn) {

    if (exportChangeReport) {
      
      Export.image.toAsset({
        image: changeReportWithMetadata,
        description: aoi_friendly_name + "_change_report_fractionalyear",
        assetId: project_asset_path + aoi_friendly_name + "_" + "change_report_fractionalyear",
        region: aoi,
        scale: 10,
        crs: CRS,
        maxPixels: 1e13
      });
      
      Export.image.toDrive({
        image: changeReportWithMetadata.toDouble(),
        description: aoi_friendly_name + "_change_report_Drive_fractionalyear",
        fileNamePrefix: aoi_friendly_name + "_" + "change_report_fractionalyear",
        region: aoi,
        scale: 10,
        crs: CRS,
        maxPixels: 1e13,
        fileFormat: "GeoTIFF"
      });
    }
  }
  
});

// ==============================================================================
// 9. EXPORTING CLASSIFIED BASELINE AND BASELINE IMAGE TO DRIVE FOR MANUAL INSPECTION OF CHANGE
// ==============================================================================


if (researchModeOn) {
  
  if (exportBaseline) {
    // assign metadata to the assets
    var baselineImageWithMetadata = baselineImage.set(pipelineParams).set("visParamsRGB", JSON.stringify(visParamsRGB));
    var baseline_filename = aoi_friendly_name + "_baseline_" + BASELINE_START + "_" + BASELINE_END
 
    Export.image.toDrive({
      image: baselineImageWithMetadata
        .visualize(visParamsRGB),
      description: baseline_filename,
      fileNamePrefix: baseline_filename,
      region: aoi,
      scale: 10,
      crs: CRS,
      maxPixels: 1e13,
      fileFormat: "GeoTIFF"
    });
  
    Export.image.toDrive({
      image: classifiedBaselineImage
        .select("classification")
        .visualize(visClassParams),
      description: baseline_filename + "_classified_Drive",
      fileNamePrefix: baseline_filename + "_classified",
      region: aoi,
      scale: 10,
      maxPixels: 1e13,
      fileFormat: "GeoTIFF"
    });
  }
}

// ==============================================================================
// 9. EXPORTING MONITORING COLLECTION TO DRIVE FOR MANUAL INSPECTION OF CHANGE
// ==============================================================================

if (researchModeOn) {
  
  var exportCollectionToDrive = function(collection, region, visParams, taskString) {
  // prepare the collection for export
  var toExport = collection.map(function(img) {
    var dateStr = img.date().format("YYYY-MM-dd");
    var visual = img.visualize(visParams);
    return visual.set("date_str", dateStr);
  });
  
  // convert collection to an ee.List to allow indexing
  var size = toExport.size();
  var collectionList = toExport.toList(size);
  
  // get an ee.List of all the date strings
  var datesList = toExport.aggregate_array("date_str");
  
  // evaluate the dates list to bring it to the client side
  datesList.evaluate(function(dates, error) {
    if (error) {
      print("Error evaluating dates:", error);
      return;
    }
    
    for (var i = 0; i < dates.length; i++) {
      var dateString = dates[i];
      var safeDate = dateString.replace(/-/g, "_");
      var taskName = "Export_Quicklook_" + safeDate;
      
      // fetch the specific image from the server-side list using the index
      var img = ee.Image(collectionList.get(i));
      
      // create the export task
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
  
  if (exportTimeSeriesRGBQuicklooks) {
    exportCollectionToDrive(monitoringImages, aoi, visParamsRGB, "S2_Quicklook_RGB_");
  }
  
  if (exportTimeSeriesClfQuicklooks) {
    exportCollectionToDrive(classifiedMonitoringCollection, aoi, visClassParams, "S2_Quicklook_Classified_")
  }
  
}