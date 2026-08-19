var pyeo = require('users/mp730/A4F:pyeoChangeAlertsSEPAL')
var cloudMasking = require('users/mp730/A4F:cloudMasking');
var project_asset_path = 'projects/aim4forests-499914/assets/'
var aoi_friendly_name = 'Kenya_II'

// ==============================================================================
// 1. PARAMETERS & CONSTANTS 
// ==============================================================================

var inspection_marker = ee.Geometry.Point(34.92526, -1.22537);

// construct a roughly 30 km2 square Area Of Interest
var corner_coordinate = [34.92372, -1.23205]
var lon = corner_coordinate[0]
var lat = corner_coordinate[1]
var aoi = ee.Geometry.Rectangle([lon - 0.025, lat - 0.025, lon + 0.025, lat + 0.025]);
var CRS = "EPSG:32737";

var BASELINE_START = '2020-01-01';
var BASELINE_END = '2020-12-31';
var MONITORING_START = '2021-01-01';
var MONITORING_END = '2021-12-31';

var BANDS = ['B2', 'B3', 'B4', 'B6', 'B8', 'B11', 'B12'];
var MAX_CLOUD_PROBABILITY_PER_PIXEL = 30; // 100 = minimal discrimination
var MAX_CLOUD_SCORE_PER_PIXEL = 30; // 100 = no discrimination 

var FOREST = 0;
var SOIL = 1;
var GRASSLAND = 2;
var URBAN = 3;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, GRASSLAND, URBAN];
var allClasses = [FOREST, SOIL, GRASSLAND, URBAN];
var changeFromClassesStr = ["Forest"];
var changeToClassesStr = ["Soil", "Grassland", "Urban"];
var allClassesStr = ["Forest", "Soil", "Grassland", "Urban"];


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
  2: "LightGreen", // grassland
  3: "Maroon" // brown forest
}

var classNameMap = {
  0: allClassesStr[0],
  1: allClassesStr[1],
  2: allClassesStr[2],
  3: allClassesStr[3]
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
    ee.Feature(ee.Geometry.Point([34.931987, -1.209981]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.933564, -1.209498]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.924444, -1.209766]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.92497, -1.212115]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.922996, -1.214647]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.920871, -1.209927]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.93351, -1.211815]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.924819, -1.220549]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.92189, -1.217728]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.917116, -1.220709]), {'class': FOREST}),
        ee.Feature(ee.Geometry.Point([34.923038, -1.223681]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.931686, -1.222769]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.906173, -1.213947]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.900755, -1.212928]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.903802, -1.21871]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.904588, -1.220663]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.902121, -1.222701]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.899986, -1.220449]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.906562, -1.216587]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.899048, -1.230646]), {'class': FOREST}), 
        ee.Feature(ee.Geometry.Point([34.913886, -1.228308]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.903501, -1.229541]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.902514, -1.226656]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.902428, -1.230753]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.901221, -1.239458]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.905298, -1.237914]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.906564, -1.240327]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.905545, -1.242569]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.900706, -1.239598]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.904191, -1.243501]), {'class': FOREST}), 
        ee.Feature(ee.Geometry.Point([34.900318, -1.242193]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.900383, -1.250515]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.89916, -1.247351]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.914111, -1.252523]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.904852, -1.252051]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.928938, -1.253192]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.934431, -1.254371]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.922297, -1.255122]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.936822, -1.254694]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.939574, -1.249642]), {'class': FOREST}), 
        ee.Feature(ee.Geometry.Point([34.94011, -1.251658]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.931205, -1.250232]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.94153, -1.247217]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.94682, -1.244568]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.942764, -1.245222]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.948332, -1.244783]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.937711, -1.245576]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.946964, -1.240914]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.947115, -1.242834]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.948005, -1.234703]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([34.936106, -1.210431]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.936815, -1.210442]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.932941, -1.214422]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.932652, -1.214625]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.927008, -1.209745]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.927405, -1.218253]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.92527, -1.218467]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.924884, -1.216783]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.925161, -1.21865]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.921674, -1.21687]), {'class': SOIL}),
        ee.Feature(ee.Geometry.Point([34.90905, -1.211155]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.901787, -1.21493]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.905574, -1.211702]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.90905, -1.211144]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.911143, -1.214437]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.900766, -1.209728]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.903599, -1.211434]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.909985, -1.223823]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.913429, -1.224874]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.902797, -1.221474]), {'class': SOIL}), 
        ee.Feature(ee.Geometry.Point([34.904138, -1.222836]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.905329, -1.225271]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.901619, -1.226406]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.911382, -1.228788]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.909938, -1.230946]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.906644, -1.230056]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.901666, -1.234215]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.909756, -1.231019]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.900711, -1.233561]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.911086, -1.230622]), {'class': SOIL}), 
        ee.Feature(ee.Geometry.Point([34.904177, -1.234998]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.903013, -1.239018]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.901801, -1.242966]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.89914, -1.241228]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.901187, -1.24422]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.901434, -1.243566]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.905286, -1.246182]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.905286, -1.248134]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.911365, -1.25441]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.920106, -1.256082]), {'class': SOIL}), 
        ee.Feature(ee.Geometry.Point([34.936597, -1.253063]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.937369, -1.253128]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.932595, -1.256539]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.942449, -1.251605]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.947088, -1.248687]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.935608, -1.244439]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.934873, -1.237374]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.941932, -1.234896]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.936225, -1.237739]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.941922, -1.234853]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([34.930442, -1.211933]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.932169, -1.211826]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.925785, -1.211579]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.933585, -1.213746]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.928757, -1.214196]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.926998, -1.213091]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.924648, -1.21514]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.923629, -1.210753]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.92232, -1.211365]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.920324, -1.220645]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.917813, -1.220334]), {'class': GRASSLAND}),
        ee.Feature(ee.Geometry.Point([34.90303, -1.212742]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.899897, -1.213804]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.89965, -1.2102]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.907321, -1.215481]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.902644, -1.21857]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.902668, -1.224145]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.909663, -1.221034]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.90962, -1.226086]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.901187, -1.220123]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.908736, -1.229745]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.905378, -1.227836]), {'class': GRASSLAND}),
        ee.Feature(ee.Geometry.Point([34.908382, -1.233647]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.902213, -1.234698]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.901366, -1.235524]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.904649, -1.231008]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.912486, -1.239834]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.901972, -1.237452]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.899451, -1.239254]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.910566, -1.242569]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.908923, -1.249593]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.918175, -1.255203]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.923357, -1.256243]), {'class': GRASSLAND}),
        ee.Feature(ee.Geometry.Point([34.9275, -1.254565]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.932531, -1.25553]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.940448, -1.253707]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.942421, -1.250049]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.948075, -1.245759]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.945071, -1.248462]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.948075, -1.250682]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.948336, -1.244332]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.938948, -1.243689]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.943959, -1.242187]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.938519, -1.242434]), {'class': GRASSLAND}),
        ee.Feature(ee.Geometry.Point([34.936513, -1.24445]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.945031, -1.241854]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.942296, -1.241768]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.944581, -1.24017]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.94513, -1.239112]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.946632, -1.237513]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.947501, -1.238618]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.946342, -1.237556]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.942555, -1.235604]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.947683, -1.23716]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.939647, -1.240013]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([34.91615, -1.220495]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.914181, -1.22016]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.914181, -1.221565]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.914074, -1.220192]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.922372, -1.210983]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.928037, -1.212624]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.906841, -1.219033]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.908451, -1.221854]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.914105, -1.220213]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.914212, -1.221564]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.906562, -1.219505]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.900717, -1.230171]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.913667, -1.23015]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.900717, -1.230182]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.90702, -1.230346]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.909446, -1.238194]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.899138, -1.243105]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.911258, -1.254839]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.946385, -1.239444]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.946943, -1.239144]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.936139, -1.232023]), {'class': URBAN}),
    ee.Feature(ee.Geometry.Point([34.946835, -1.227624]), {'class': URBAN})
])

if (researchModeOn) {
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', FOREST)),
    {color: classColourMap[1]}, 'Training: Forest')
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', SOIL)),
    {color: classColourMap[2]}, 'Training: Soil')
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', GRASSLAND)),
    {color: classColourMap[3]}, 'Training: Grassland')
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', URBAN)),
    {color: classColourMap[4]}, 'Training: Urban')
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