var pyeo = require('users/mp730/A4F:pyeoChangeAlerts')
var cloudMasking = require('users/mp730/A4F:cloudMasking');

// ==============================================================================
// 1. PARAMETERS & CONSTANTS 
// ==============================================================================

// L9 - binary change detection value of 0
// var inspection_marker = ee.Geometry.Point(35.30557, -0.39567);

// L9 - binary change detection value of 1
//var inspection_marker = ee.Geometry.Point(35.305064, -0.395502); 

// L9 - binary change detection value of masked
var inspection_marker = ee.Geometry.Point(35.304702, -0.39817);

// construct a roughly 30 km2 square Area Of Interest
var corner_coordinate = [35.2745, -0.4285]
var lon = corner_coordinate[0]
var lat = corner_coordinate[1]
var aoi = ee.Geometry.Rectangle([lon, lat, lon + 0.05, lat + 0.05]);

print("AOI area (km2)", aoi.area().divide(1000 * 1000))

var BASELINE_START = '2020-01-01';
var BASELINE_END = '2020-12-31';
var MONITORING_START = '2024-06-30';
var MONITORING_END = '2024-12-31';

var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
var MAX_CLOUD_PROBABILITY_PER_PIXEL = 30; // 100 = minimal discrimination
var MAX_CLOUD_SCORE_PER_PIXEL = 30; // 100 = no discrimination 

var FOREST = 1;
var SOIL = 2;
var AGRICULTURE = 3;
var URBAN = 4;
var changeFromClasses = [FOREST];
var changeToClasses = [SOIL, AGRICULTURE];
var allClasses = [FOREST, SOIL, AGRICULTURE, URBAN];

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

// DEV PARAMS - hardcoded first and last change date params, only works for the aoi, classifier and time range of this test
var minDate = 1720080614769
var maxDate = 1735200612936

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
var finalImage = ee.Image(imageList.get(35))

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
//     {colour: 'green'}, 'Training: Forest')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', SOIL)),
//     {colour: 'brown'}, 'Training: Soil')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', AGRICULTURE)),
//     {colour: 'orange'}, 'Training: Agriculture')
// Map.addLayer(
//     trainingPoints.filter(ee.Filter.eq('class', URBAN)),
//     {colour: 'blue'}, 'Training: Urban')

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
// 8. VISUALISING CHANGE REPORT LAYER FOR MID TERM REPORT
// ==============================================================================

// get minMax date stats for first date of change colour ramp
var dateStats = alerts.changeReport.select("first_change_date_above_threshold").reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: aoi,
    scale: 10,
    maxPixels: 1e9
});
var dateVisPalette = ['#ffffb2', '#fecc5c', '#fd8d3c']; // pale yellow to orange

// first and last change date visual parameters
var dateVisParams = {
  min: minDate,
  max: maxDate,
  palette: dateVisPalette
}

var minVal = ee.Number(dateStats.get("first_change_date_above_threshold_min"));
var maxVal = ee.Number(dateStats.get("first_change_date_above_threshold_max"));

// chuck into one dictionary to evaluate in one go
var serverValues = ee.Dictionary({
  start: minVal,
  end: maxVal
});

// create separate map instances for a multi "panel" visualisation
var mapBaseline = ui.Map();
var mapFinalChangeImage = ui.Map();
var mapChangeReportDates = ui.Map();

// label map instance titles
mapBaseline.add(ui.Label("Baseline Image", {position: "top-center"})); 
mapFinalChangeImage.add(ui.Label("Final Image of Change Period", {position: "top-center"}));
mapChangeReportDates.add(ui.Label("Dates of Detected Changes", {position: "top-center"}))

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
  );

mapChangeReportDates.addLayer(
  finalImage,
  visParamsRGB,
  "Final Image of Monitoring Stack"
  );

mapChangeReportDates.addLayer(
  alerts.changeReport.select("fcd_decision_map"),
  dateVisParams,
  "First Dates of Detected Changes"
  );
  
// synchronise the maps together
var linker = ui.Map.Linker([mapBaseline, mapFinalChangeImage, mapChangeReportDates]);

// create a nested layout:
// 2 on the top row
var topRow = ui.Panel(
  [mapBaseline, mapFinalChangeImage],
  ui.Panel.Layout.Flow("horizontal"),
  {stretch: "both"}
);

//  1 on the bottom row
var mapGrid = ui.Panel(
  [topRow, mapChangeReportDates],
  ui.Panel.Layout.Flow("vertical"),
  {stretch: "both"}
)
// replace default map instance of the code editor with the new grid
ui.root.widgets().reset([mapGrid]);

// center the map
mapBaseline.centerObject(aoi, 14);

// evaluate, so populates the relevant ui panel later instead of pausing rendering until completed
serverValues.evaluate(function(clientValues) {
  
  // get the start and end values
  var startMillis = clientValues.start;
  var endMillis = clientValues.end;
  
  // parse date function
  function parseDate(millis) {
    var date = new Date(millis) // convert to JS Date object
    return date.toISOString().split("T")[0] // .toISOString() returns this format "2011-10-05T14:48:00.000Z"
  }
  
  // function to make parameters needed for a colour bar
  var makeColourBarParams = function(palette) {
    return {
      bbox: [0, 0, 1, 0.1],
      dimensions: '100x10',
      format: 'png',
      min: 0,
      max: 1,
      palette: palette,
    };
  };

  // make the colour bar from ui.Thumbnail
  var colourBar = ui.Thumbnail({
    image: ee.Image.pixelLonLat().select(0),
    params: makeColourBarParams(dateVisPalette),
    style: {stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px'},
  });

  // make the date labels
  var legendLabels = ui.Panel({
    widgets: [
      ui.Label(parseDate(startMillis), {margin: '4px 8px'}),
      ui.Label('', {margin: '4px 8px', textAlign: 'center', stretch: 'horizontal'}),
      ui.Label(parseDate(endMillis), {margin: '4px 8px'})
    ],
    layout: ui.Panel.Layout.flow('horizontal')
  });

  var legendTitle = ui.Label({
    value: 'First Date of Detected Change',
    style: {fontWeight: 'bold', margin: '8px 8px 4px 8px'}
  });

  // bring the individual colour bar components together as a panel
  var legendPanel = ui.Panel(
    [legendTitle, colourBar, legendLabels],
    ui.Panel.Layout.flow('vertical'),
    {
      position: 'bottom-left',
      padding: '8px',
      backgroundColor: 'rgba(255, 255, 255, 0.9)'
    }
  );

  mapChangeReportDates.add(legendPanel);

})
