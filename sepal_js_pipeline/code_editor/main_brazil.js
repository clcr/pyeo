var pyeo = require('users/mp730/A4F:pyeoChangeAlertsSEPAL')
var cloudMasking = require('users/mp730/A4F:cloudMasking');
var project_asset_path = 'projects/aim4forests-499914/assets/'
var aoi_friendly_name = 'Brazil_II'

// ==============================================================================
// 1. PARAMETERS & CONSTANTS 
// ==============================================================================

var inspection_marker = ee.Geometry.Point(-59.16421, -15.0234);

// construct a roughly 30 km2 square Area Of Interest
var corner_coordinate = [-59.161552060067635, -15.02035785032475] 
var lon = corner_coordinate[0]
var lat = corner_coordinate[1]
var aoi = ee.Geometry.Rectangle([lon - 0.025, lat - 0.025, lon + 0.025, lat + 0.025]);
var CRS = "EPSG:5641"

var BASELINE_START = '2019-07-01';
var BASELINE_END = '2020-06-30';
var MONITORING_START = '2020-09-01';
var MONITORING_END = '2021-12-31';

var BANDS = ['B2', 'B3', 'B4', 'B6', 'B8', 'B11', 'B12'];
var MAX_CLOUD_PROBABILITY_PER_PIXEL = 30; // 100 = minimal discrimination
var MAX_CLOUD_SCORE_PER_PIXEL = 30; // 100 = no discrimination 

// class integers must be indexed from 0 for the confusion matrix export to work
var FOREST = 0;
var SOIL = 1;
var GRASSLAND = 2;
var BROWN_FOREST = 3;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, GRASSLAND];
var allClasses = [FOREST, SOIL, GRASSLAND, BROWN_FOREST];

var changeFromClassesStr = ["Forest"];
var changeToClassesStr = ["Soil", "Grassland"];
var allClassesStr = ["Forest", "Soil", "Grassland", "Brown Forest"];

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

// temporarily remove problematic images - problematic because poor cloud masking / haze
var cleanedMonitoringCollection = maskedMonitoringCollection
  // .filter(ee.Filter.neq("system:index", "20200906T141049_20200906T141049_T21LTD"))
  // .filter(ee.Filter.neq("system:index", "20201220T141051_20201220T141045_T21LTD"))
  // .filter(ee.Filter.neq("system:index", "20210114T141049_20210114T141046_T21LTD"));
  
var baselineImage = maskedBaselineCollection
  .map(addNDVI)
  .select(BANDS.concat("gate_index"))
  .median()
  .clip(aoi);

var monitoringImagesRaw = cleanedMonitoringCollection
  .map(addNDVI)
  .select(BANDS.concat("gate_index"))
  
var monitoringImages = dailyMosaic(monitoringImagesRaw)

Map.addLayer(
  baselineImage,
  visParamsRGB,
  'Baseline Image', false
)

var trainingPoints = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([-59.163997, -15.014472]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.157452, -15.01957]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.156873, -15.01584]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.163177, -15.027786]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.160152, -15.022647]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.166697, -15.024864]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.180472, -15.029406]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.165023, -15.034794]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.171653, -15.041881]), {'class': FOREST}), 
    ee.Feature(ee.Geometry.Point([-59.182039, -15.039415]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15563, -15.00212]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16844, -15.00104]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.14668, -15.00059]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.13802, -15.00279]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.17419, -15.00067]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16522, -15.02024]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.14059, -15.00859]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.17037, -14.9998]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.13844, -15.01825]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.18449, -15.0122]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16604, -15.02177]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.17286, -15.00701]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.14559, -15.03389]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.1855, -15.03029]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.14362, -15.03961]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.17563, -15.02515]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.1458, -15.04341]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.1652, -15.03956]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.13731, -15.04179]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.18233, -15.04407]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.18593, -15.04353]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.13679, -15.03674]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16525, -15.03947]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.1392, -15.02754]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15722, -15.02625]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15919, -15.03549]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.14962, -15.02824]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16194, -15.00384]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16808, -15.00127]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15408, -15.01325]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.14082, -15.00607]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16636, -15.00972]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15327, -15.02386]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.13945, -15.00284]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16241, -15.022364]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15902, -15.017453]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.174577, -15.014841]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.162603, -15.012748]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.145244, -15.016935]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.165286, -15.017743]), {'class': FOREST}),
    ee.Feature(ee.Geometry.Point([-59.18412, -15.033261]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.179399, -15.039602]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.181786, -15.035011]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.179791, -15.036317]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.144097, -15.033533]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.16976, -15.032642]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.163559, -15.030901]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.160598, -15.028414]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.145022, -15.019395]), {'class': SOIL}), 
    ee.Feature(ee.Geometry.Point([-59.145966, -15.024431]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.167346, -15.019256]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.177904, -15.002563]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.177442, -15.003568]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.180479, -15.006273]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.183, -15.006035]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.178075, -15.006211]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.169688, -15.006815]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.168615, -15.009924]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.174537, -15.009281]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.16876, -15.013171]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.179424, -15.015181]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.179607, -15.009046]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.179418, -15.015202]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.182691, -15.017585]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.17435, -15.020339]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.183137, -15.022173]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.180863, -15.021759]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.17126, -15.019023]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.182418, -15.01721]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.183706, -15.022966]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.186367, -15.023225]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.171455, -15.02599]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.179587, -15.028829]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.168779, -15.013177]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.172856, -15.01099]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.171708, -15.012493]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.17203, -15.016358]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.174634, -15.019269]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.166673, -15.019559]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.170632, -15.022492]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.171962, -15.016295]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.175707, -15.022637]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.170062, -15.025189]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.175705, -15.022608]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.169892, -15.028846]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.176114, -15.037985]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.180921, -15.031996]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.1812, -15.027023]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.183152, -15.037799]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.139819, -15.044919]), {'class': SOIL}),
    ee.Feature(ee.Geometry.Point([-59.14116, -15.017405]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.152219, -15.025272]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.138615, -15.021003]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.145889, -15.022164]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.137896, -15.033297]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.14399, -15.033048]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.147757, -15.044781]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.158572, -15.042087]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.15844, -15.028466]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.172623, -15.027513]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.173224, -15.033999]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.142449, -15.005888]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.141419, -14.998862]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.138308, -15.004313]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.152556, -15.002075]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.147921, -15.003754]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.145836, -15.009684]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.152209, -15.002098]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.143755, -15.013477]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.150509, -15.01707]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.141539, -15.0162]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.151818, -15.023743]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.158169, -15.025301]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.170658, -15.021197]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.171881, -15.027415]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.173855, -15.027207]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.177331, -15.029238]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.182116, -15.025943]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.182781, -15.021011]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.17055, -15.021073]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.182803, -15.021218]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.177503, -15.03535]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.171259, -15.036283]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.172654, -15.042002]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.165079, -15.040345]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.171967, -15.035413]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.174821, -15.044945]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.163363, -15.035786]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.172804, -15.030714]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.17465, -15.030548]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.158122, -15.034481]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.15132, -15.029819]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.161899, -15.033466]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.164474, -15.031373]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.14308, -15.036118]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.160032, -15.030772]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.164602, -15.042356]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.168164, -15.033051]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.148917, -15.038564]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.164559, -15.04248]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.14167, -15.028619]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.142636, -15.026878]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.138537, -15.032452]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.137057, -15.03784]), {'class': GRASSLAND}),
    ee.Feature(ee.Geometry.Point([-59.160092, -15.032196]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.166323, -15.01849]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.155251, -15.023464]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.139674, -15.0167]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.138344, -15.017031]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15647, -15.000537]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.14514, -14.996101]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.154496, -15.010071]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.156169, -15.007729]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.157907, -15.00605]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.157736, -15.006318]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15559, -15.008224]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.1784, -15.030623]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.152114, -15.029255]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.183983, -15.021827]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.182781, -15.022884]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.178387, -15.030615]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.176714, -15.028978]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.16047, -15.030905]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.180812, -15.025144]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.175169, -15.028356]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.184254, -15.021957]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.185509, -15.014906]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.184715, -15.007154]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.156648, -15.00744]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.158214, -15.005429]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.155981, -15.007854]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15803, -15.0062]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.157923, -15.00594]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.157408, -15.006562]), {'class': BROWN_FOREST})
])

if (researchModeOn) {
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', FOREST)),
    {color: classColourMap[0]}, 'Training: Forest')
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', SOIL)),
    {color: classColourMap[1]}, 'Training: Soil')
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', GRASSLAND)),
    {color: classColourMap[2]}, 'Training: Grassland')
  Map.addLayer(
    trainingPoints.filter(ee.Filter.eq('class', BROWN_FOREST)),
    {color: classColourMap[3]}, 'Training: Brown Forest')
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