/**
 * PyEO Change Alerts — GEE function.
 *
 * The input parameters and output band spec below are the contract between
 * UoL and SEPAL. Implementation strategy is up to you (per-image .map(),
 * ImageCollection.iterate with running state, two-pass, ...).
 *
 * This function does not classify. It receives baseline and monitoring
 * inputs that already carry a 'class' band; the SEPAL recipe wrapper
 * applies the user's classification recipe before calling in.
 *
 * @param {Object} params
 * @param {ee.Geometry|ee.FeatureCollection} params.aoi
 *     Area of interest. Output should be clipped to this.
 * @param {ee.Image} params.classifiedBaseline
 *     Baseline classification map with a 'class' band (integer class IDs).
 *     If params.dNdviGate is set, must also carry params.dNdviGate.band
 *     (typically 'NDVI').
 * @param {ee.ImageCollection} params.classifiedMonitoringCollection
 *     Monitoring images, time-sorted (system:time_start ascending). Each
 *     image carries the same 'class' band (and optional gate band) as the
 *     baseline.
 * @param {number[]} params.changeFromClasses
 *     Class IDs that, in the BASELINE, count as the "from" side of a change.
 * @param {number[]} params.changeToClasses
 *     Class IDs that, in any MONITORING image, count as the "to" side.
 * @param {number} [params.minConsecutiveDetections=2]
 *     Temporal-confidence parameter. Interpretation up to the algorithm.
 * @param {{band: string, threshold: number}} [params.dNdviGate]
 *     Optional spectral confirmation: alert is confirmed only when `band`
 *     drops by at least `threshold` vs. the baseline.
 *
 * @returns {ee.Image} Multi-band alert raster, clipped to AOI. Required bands:
 *     - first_change_date   days since 2000-01-01, masked where no alert
 *     - detection_count     total monitoring images flagged at this pixel
 *     - consecutive_count   longest consecutive run of detections
 *     - from_class          baseline class ID at the alert pixel
 *     - to_class            most recent monitoring class ID at the alert pixel
 *     - confidence          [0..1]
 *     - dndvi_drop          (only when dNdviGate is set) index change at confirmation
 *
 *     Pixels with no alert should be masked out. The output band names are
 *     part of the contract — SEPAL's wrapper references them. If the
 *     algorithm needs a different schema, flag it so we update both sides.
 *
 *     Keep the body server-side. Use .map() / .iterate() / ee.Algorithms.If.
 *     Avoid .getInfo() and JS for-loops over ee.List — they fail at scale.
 */
 // 
 // potential inputs: dNDVI_boolean (on/off), dNDVI threshold
 //
var run_change_detection = function (params) {
  // input parameters
  var aoi = params.aoi
  var classifiedBaseline = params.classifiedBaseline
  var classifiedMonitoringCollection = params.classifiedMonitoringCollection
  var changeFromClasses = params.changeFromClasses
  var changeToClasses = params.changeToClasses
  var minRequiredValidatedDetectionsThreshold = params.minRequiredValidatedDetectionsThreshold || 2
  var dNdviGate = params.dNdviGate || {use_ndvi: false,  band: 'NDVI', threshold: -2.0}
  var PercentageProbabilityThreshold = params.PercentageProbabilityThreshold || 50
  
  if (!(dNdviGate.use_ndvi)) {
    dNdviGate.threshold = -2.0
  }

  // start of run_change_detection function
  var baselineClassification = classifiedBaseline.select("classification");
  var baselineNDVI = classifiedBaseline.select("NDVI");
  
  // create a fromMask that works for multiple classes
  // this mask is used to count changes for pixels from a FROM class to a TO class
  var fromMask = ee.ImageCollection(
    changeFromClasses.map(function(classId) {
      return baselineClassification.eq(classId)
    })
    ).max(); // max() works as a logical OR across the imagecollection

  // ************
  // set up the logic to map over the classified monitoring collection 
  // ************
  // to locate the valid change event pixels per image
  // concats 3 bands to each image of the monitoring collection
  // e.g. 38 images with 2 bands each, becomes 38 images with 5 bands each
  
  var changeEvents = classifiedMonitoringCollection.map(function(image) {
    var currentClass = image.select("classification");
    var currentNDVI = image.select("NDVI");

    // ************
    // for each image, identify where pixels are a FROM class in the monitoring collection
    // this describes the consistency of each pixel as a FROM class
    var isFromClass = ee.ImageCollection(
      changeFromClasses.map(function(classId) {
        return currentClass.eq(classId);
      })
    ).max().rename("is_from_class");
    // ************

    // ************
    // for each image, identify where pixels are a TO class in the monitoring collection
    // this describes the consistency of each pixel as a TO class
    var isToClass = ee.ImageCollection(
      changeToClasses.map(function(classId) {
        return currentClass.eq(classId);
      })
    ).max().rename("is_to_class");
    // ************

    // identify which pixels have changed to ChangeToClasses
    var transitionMask = fromMask.and(isToClass); // using a mask allows for .and to work
      
    // calculate whether the NDVI is greater than the delta NDVI
    var deltaNDVI = baselineNDVI.subtract(currentNDVI).rename("delta_ndvi");
    var ndviMask = deltaNDVI.gte(dNdviGate.threshold);
    var deltaNDVIthresholded = deltaNDVI.updateMask(ndviMask).rename("delta_ndvi_thresholded");

    // flag where both conditions (class and NDVI change) are met
    var isChangeMask = transitionMask.and(ndviMask).rename("is_change");
      
    // store the image_date as a band for change reporting
    var imgMillis = ee.Image.constant(image.getNumber("system:time_start"));
    var imgMillisGeneric = imgMillis.double();
    
    // build dates of all changes above the NDVI threshold
    var changeDateAboveThreshold = imgMillisGeneric.updateMask(isChangeMask).rename("change_date_above_threshold")

    return image.addBands([isChangeMask, changeDateAboveThreshold, deltaNDVI, deltaNDVIthresholded, isFromClass, isToClass]);
  }); // end of changeEvents function
  // an imagecollection, each image has the six bands above
  
  // set up an image that tracks change persistency
  // var initialState = ee.Image([
  //   ee.Image.constant(0).rename("streak"),
  //   ee.Image.constant(0).rename("max_streak"),
  //   ee.Image.constant(0).rename("first_date"),
  //   ee.Image.constant(0).rename("last_date")
  // ]); 
  
  // track the change streaks across the timeseries
  // var calculateStreaks = function(image, state) {
  //   state = ee.Image(state);
  //   var isChange = image.select("is_change");
  //   var currentDate = image.select("image_date");
    
  //   // create a mask of valid (unclouded) pixels in the image to ensure this does not break a valid streak
  //   var isValid = isChange.mask();
    
  //   // get the current streak, append + 1 if there is a change
  //   var currentStreak = state.select("streak");
  //   var newStreak = currentStreak.add(1).multiply(isChange.unmask(0));

  //   // if pixel is cloud masked, keep the old streak
  //   // https://developers.google.com/earth-engine/apidocs/ee-image-where
  //   // "For each pixel in 'currentStreak', if the corresponding pixel in 'isValid' is 1, 
  //   //    output the corresponding pixel in newStreak, otherwise output the input pixel."
  //   var updatedStreak = currentStreak.where(isValid, newStreak);

  //   var maxStreak = state.select("max_streak");
  //   var updatedMaxSteak = maxStreak.max(updatedStreak);

  //   // 6: if streak progresses from 0 to 1 (a change), get image_date
  //   var firstDate = state.select("first_date");
  //   var isFirstChange = currentStreak.eq(0).and(updatedStreak.eq(1));
  //   var updatedFirstDate = firstDate.where(isFirstChange.and(isValid), currentDate);

  //   // 7: update last_date every time a valid change occurs
  //   var lastDate = state.select("last_date");
  //   var updatedLastDate = lastDate.where(isChange.unmask(0).eq(1).and(isValid), currentDate);

  //   return ee.Image([
  //       updatedStreak,
  //       updatedMaxSteak,
  //       updatedFirstDate,
  //       updatedLastDate
  //   ]);
  // }; // end of calculateStreaks function

  // // run calculateStreaks across the monitoringCollection
  // var finalState = ee.Image(changeEvents.iterate(calculateStreaks, initialState));

  // // filter the final output to only include pixels that met the consecutive threshold
  // var validPixels = finalState.select("max_streak").gte(minRequiredValidatedDetectionsThreshold);
  
  // // update finalState with the pixels that met the minConsecutiveDetections threshold
  // finalState = finalState.updateMask(validPixels);
  
  // LAYER 0: count the number of images within the collection
  var availableImageCount = ee.Image(
    ee.Image.constant(classifiedMonitoringCollection.size()))
    .clip(aoi)
    .rename("available_image_count");
    
  // LAYER 1: count the number of occluded images per pixel
  var occludedCount = classifiedMonitoringCollection.map(function(image) {
    var isOccluded = image.select("classification").mask().unmask(0).eq(0);
    // .mask() returns a 1 for valid pixels and is masked for cloudy pixels
    // unmask(0) converts these cloudy pixels to 0, but it also respects the image footprint
    // and leaves pixels outside of the image boundary fully masked
    // .eq(0) turns the 0s (clouds) into 1s so these can be summed and counted
    return isOccluded.rename("occluded_count");
  }).sum();

  // LAYER 2: class change detection count - how many times a pixel changed from a FROM class to a TO class
  var classChangeDetectionCount = changeEvents.select("is_change").sum().rename("total_changes")

  // LAYER 3: firstChangeDate (FCD) that passes the dNDVI threshold if present
  var firstChangeDateAboveThreshold = changeEvents.select("change_date_above_threshold").min().rename("first_change_date_above_threshold");

  // LAYER 15: count the number of pixels that were a FROM class
  // "counts" by summing across the collection https://developers.google.com/earth-engine/apidocs/ee-imagecollection-sum
  var fromClassCount = changeEvents.select("is_from_class").sum().rename("from_class_count");
  // ************

  // LAYER 16: count the number of pixels that were a TO class
  var toClassCount = changeEvents.select("is_to_class").sum().rename("to_class_count");

  // concat all additional bands together
  // var finalOutput = finalState.addBands([
  //   fromClassCount
  // ]);

  return {
    changeReport: ee.Image([
      availableImageCount,
      occludedCount,
      classChangeDetectionCount,
      firstChangeDateAboveThreshold,
      fromClassCount,
      toClassCount]
    ),
    changeEvents: changeEvents, // an imagecollection
    fromClassCollection: changeEvents.select("is_from_class"), // an imagecollection
    toClassCollection: changeEvents.select("is_to_class") // an imagecollection
  };

}

exports.run_change_detection = run_change_detection;