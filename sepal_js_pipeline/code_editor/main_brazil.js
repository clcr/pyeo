var pyeo = require('users/mp730/A4F:pyeoChangeAlerts')
var cloudMasking = require('users/mp730/A4F:cloudMasking');
var project_asset_path = 'projects/aim4forests-499914/assets/'
var aoi_friendly_name = 'Brazil_II'

// ==============================================================================
// 1. PARAMETERS & CONSTANTS 
// ==============================================================================

var inspection_marker = ee.Geometry.Point(-59.16104, -15.01913);

// construct a roughly 30 km2 square Area Of Interest
var corner_coordinate = [-59.161552060067635, -15.02035785032475] 
var lon = corner_coordinate[0]
var lat = corner_coordinate[1]
var aoi = ee.Geometry.Rectangle([lon - 0.025, lat - 0.025, lon + 0.025, lat + 0.025]);
var CRS = "EPSG:5641";

print("AOI area (km2)", aoi.area().divide(1000 * 1000))

var BASELINE_START = '2020-01-01';
var BASELINE_END = '2020-12-31';
var MONITORING_START = '2021-01-01';
var MONITORING_END = '2021-10-12' // '2022-03-31'; // subtract 45, 35, 30 worked with this end date

var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
var MAX_CLOUD_PROBABILITY_PER_PIXEL = 30; // 100 = minimal discrimination
var MAX_CLOUD_SCORE_PER_PIXEL = 30; // 100 = no discrimination 

var FOREST = 1;
var SOIL = 2;
var GRASSLAND = 3;
var BROWN_FOREST = 4;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, GRASSLAND, BROWN_FOREST];
var allClasses = [FOREST, SOIL, GRASSLAND, BROWN_FOREST];

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
  'change_to_classes': JSON.stringify(changeToClasses),
  'all_classes': JSON.stringify(allClasses),
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
  3: "LightGreen", // grassland
  4: "Maroon" // brown forest
}

// total changes palette
var totalChangesPalette = ["#FFE2E2", "#9F0712"]; // reds

// fcd decision map palette
var fcdDecisionMapPalette = ["#DBEAFE", "#1C398E"]; // blues

// fcd repeatability palette
var fcdRepeatabilityPalette = ["#DCFCE7", "#0D542B"]; // greens

// 1625321810077
// 1624457809201
// 1610633808389

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
var listLength = monitoringImages.size();
var imageList = monitoringImages.toList(listLength);
var firstImage = ee.Image(imageList.get(1));
var finalImage = ee.Image(imageList.get(listLength.subtract(1)));

print(imageList)

Map.addLayer(
  baselineImage,
  visParamsRGB,
  'Baseline Image', false
)

Map.addLayer(
  firstImage,
  visParamsRGB,
  "Beginning Monitoring Image", false)

Map.addLayer(
  finalImage,
  visParamsRGB,
  "Ending Monitoring Image")

// Inline training: forest / non-forest points within the AOI. Test fixture
// only — disappears once the SEPAL CLASSIFICATION recipe wrapper is in place.
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
    ee.Feature(ee.Geometry.Point([-59.160092, -15.032196]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.166323, -15.01849]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.155251, -15.023464]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.139674, -15.0167]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.138344, -15.017031]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.15647, -15.000537]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.14514, -14.996101]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.154496, -15.010071]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.156169, -15.007729]), {'class': BROWN_FOREST}),
    ee.Feature(ee.Geometry.Point([-59.157907, -15.00605]), {'class': BROWN_FOREST})
])

// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', FOREST)),
//     {colour: 'green'}, 'Training: FOREST')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', SOIL)),
//     {colour: 'brown'}, 'Training: SOIL')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', GRASSLAND)),
//     {colour: 'orange'}, 'Training: GRASSLAND')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', BROWN_FOREST)),
//     {colour: 'blue'}, 'Training: BROWN FOREST')

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
  alerts.changeReport.select("total_changes"),
  totalChangesVisParams,
  "L2 - Total Changes");
  
  Map.addLayer(
  alerts.changeReport.select("post_fcd_change_repeatability_pct"),
  repeatabilityVisParams,
  "L8 - Post-FCD Change Repeatability");

  Map.addLayer(
  alerts.changeReport.select("fcd_decision_map"),
  fcdDecisionVisParams,
  "L10 - FCD Decision Map");

  var changeReportWithMetadata = alerts.changeReport
    .set(pipelineParams)
    .set("fcdDecisionVisParams", JSON.stringify(fcdDecisionVisParams))
    .set("totalChangesVisParams", JSON.stringify(totalChangesVisParams))
    .set("repeatabilityVisParams", JSON.stringify(repeatabilityVisParams));

  Export.image.toAsset({
    image: changeReportWithMetadata,
    description: aoi_friendly_name + "_change_report",
    assetId: project_asset_path + aoi_friendly_name + "_" + "change_report",
    region: aoi,
    scale: 10,
    crs: CRS,
    maxPixels: 1e13
  });

});

// assign metadata to the assets
var baselineImageWithMetadata = baselineImage.set(pipelineParams).set("visParamsRGB", JSON.stringify(visParamsRGB));
var firstImageWithMetadata = firstImage.set(pipelineParams).set("visParamsRGB", JSON.stringify(visParamsRGB));
var finalImageWithMetadata = finalImage.set(pipelineParams).set("visParamsRGB", JSON.stringify(visParamsRGB));
var baseline_filename = aoi_friendly_name + "_baseline_" + BASELINE_START + "_" + BASELINE_END
var firstImage_filename = aoi_friendly_name +  "_first_monitoring_" + MONITORING_START + "_" + MONITORING_END
var finalImage_filename = aoi_friendly_name + "_final_monitoring_" + MONITORING_START + "_" + MONITORING_END

Export.image.toAsset({
  image: baselineImageWithMetadata,
  description: baseline_filename,
  assetId: project_asset_path + baseline_filename,
  region: aoi,
  scale: 10,
  crs: CRS,
  maxPixels: 1e13
});

Export.image.toAsset({
  image: firstImageWithMetadata,
  description: firstImage_filename,
  assetId: project_asset_path + firstImage_filename,
  region: aoi,
  scale: 10,
  crs: CRS,
  maxPixels: 1e13
});

Export.image.toAsset({
  image: finalImageWithMetadata,
  description: finalImage_filename,
  assetId: project_asset_path + finalImage_filename,
  region: aoi,
  scale: 10,
  crs: CRS,
  maxPixels: 1e13
});