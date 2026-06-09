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
  // ==============================================================================
  // 1. INPUT PARAMETERS
  // ==============================================================================
  var aoi = params.aoi
  var classifiedBaseline = params.classifiedBaseline
  var classifiedMonitoringCollection = params.classifiedMonitoringCollection
  var changeFromClasses = params.changeFromClasses
  var changeToClasses = params.changeToClasses
  var minRequiredValidatedDetectionsThreshold = params.minRequiredValidatedDetectionsThreshold || 2
  var dNdviGate = params.dNdviGate || {use_ndvi: false,  band: 'NDVI', threshold: -2.0}
  var percentageProbabilityThreshold = params.percentageProbabilityThreshold || 50
  
  // if user has opted to not use NDVI (as a threshold), then set threshold to let all detections through
  if (!(dNdviGate.use_ndvi)) {
    dNdviGate.threshold = -2.0
  }

  // ==============================================================================
  // 2. CHANGE DETECTION LOGIC
  // ==============================================================================

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

  // ==============================================================================
  // 2A. changeEvents LOGIC
  // ==============================================================================

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
    var deltaNdviThresholdedMask = deltaNDVI.gte(dNdviGate.threshold).rename("delta_ndvi_thresholded_mask"); // boolean of whether a pixel passed the evaluation

    // flag where both conditions (class and NDVI change) are met
    var isChangeMask = transitionMask.and(deltaNdviThresholdedMask).rename("is_change");
      
    // store the image_date as a band for change reporting
    var imgMillis = ee.Image.constant(image.getNumber("system:time_start"));
    var imgMillisGeneric = imgMillis.double();
    
    // build dates of all changes above the NDVI threshold
    var changeDateAboveThreshold = imgMillisGeneric.updateMask(isChangeMask).rename("change_date_above_threshold")

    return image.addBands([isChangeMask, changeDateAboveThreshold, deltaNdviThresholdedMask, deltaNDVI, isFromClass, isToClass, currentClass]);
  }); // end of changeEvents function
  // an imagecollection, each image has the six bands above

  // ==============================================================================
  // 2B. postChangeEvaluation LOGIC
  // ==============================================================================

  // get the first change date per pixel
  var firstChangeDateAboveThreshold = changeEvents.select("change_date_above_threshold").min().rename("first_change_date_above_threshold");

  // map over the change images in changeEvents a second time, to evaluate temporal consistency of the first changes
  var postChangeEvaluation = changeEvents.map(function(image) {
    // get the timestamp, cast to double to ensure homogeneity of types
    var currentMillis = ee.Image.constant(image.getNumber("system:time_start")).double();

    // create a temporal window mask
    // pixel has a value of 1 if it has a change that is the first change or is afterwards
    var isAfterFirstChange = currentMillis.gte(firstChangeDateAboveThreshold);
    var isPostFCD = isAfterFirstChange.rename("post_fcd"); // boolean (1, 0) indicating whether the pixel is post-FCD
    var isChange = image.select("is_change") // 1 for change, 0 for no change
    // here find out if masked, then unmasked and eq
    // var isPostFCDOccluded = isPostFCD.mask().unmask(0).eq(0).rename("post_fcd_occluded") 

    var isOccluded = image.select("classification").mask().not();
    
    // combine first change date with whether was occluded
    // we unmask isAfterFirstChange to 0. If a pixel never had a first change, 
    // it cannot have post-FCD occlusions, so we force it to 0 instead of leaving it masked
    var isPostFCDOccluded = isOccluded.and(isAfterFirstChange.unmask(0)).rename("post_fcd_occluded");

    // a pixel had a change after after FCD
    var subsequentChange = isChange.and(isAfterFirstChange).rename("post_fcd_change")

    // a pixel did not have a new change after a FCD
    var isNotChange = isChange.not(); // turns 0s (no change) to 1s and vice versa - 1s (change) to 0s
    var subsequentNonChange = isNotChange.and(isAfterFirstChange).rename("post_fcd_nochange");

    // return an imagecollection of images with two bands each, of subsquent change and non-change
    return image.addBands([subsequentChange, subsequentNonChange, isPostFCD, isPostFCDOccluded]);
  });

  // isPostFCD summed = total number of available images post-FCD (not occluded)
  
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

  // LAYER 3: firstChangeDate (FCD) and Combined Alert Detection (changes that pass the dNDVI threshold)
  // firstChangeDateAboveThreshold computed above

  // LAYER 4: Post-FCD Combined Alert Count (count of changes that pass the dNDVI threshold since the first change date)
  // get the counts by summing the post-FCD change
  var postFCDChangeCount = postChangeEvaluation.select("post_fcd_change").sum().rename("post_fcd_change_count");

  // LAYER 5: Post-FCD Combined Non-Alert Count (count of no changes since the first change date)
  // get the counts by summing the post-FCD no changes
  var postFCDNoChangeCount = postChangeEvaluation.select("post_fcd_nochange").sum().rename("post_fcd_nochange_count");

  // LAYER 6: Post-FCD Occluded Image Count
  var postFCDOccludedCount = postChangeEvaluation.select("post_fcd_occluded").sum().rename("post_fcd_occluded_count");

  // LAYER 7: Post-FCD Valid Image Count
  var postFCDValidImageCount = postFCDChangeCount.add(postFCDNoChangeCount).rename("post_fcd_valid_image_count");

  // LAYER 8: Post-FCD Change Detection Repeatability
  var postFCDChangeDetectionRepeatability = postFCDChangeCount
    .divide(postFCDValidImageCount)
    .multiply(100)
    .rename("post_fcd_change_repeatability_pct");

  // LAYER 9: Binary time-series decision
  var binaryTimeSeriesDecision = postFCDChangeDetectionRepeatability.gte(percentageProbabilityThreshold)
    .and(postFCDChangeCount.gte(minRequiredValidatedDetectionsThreshold))
    .rename("binary_timeseries_decision")

  // LAYER 10: FCD Decision Map
  var FCDDecisionMap = firstChangeDateAboveThreshold
    .updateMask(binaryTimeSeriesDecision)
    .rename("fcd_decision_map")

  // LAYER 11: dNDVI only change detection count
  var deltaNDVIChangeDetectionCount = changeEvents
    .select("delta_ndvi_thresholded_mask")
    .sum()
    .rename("deltaNDVI_change_count");

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
      postFCDChangeCount,
      postFCDNoChangeCount,
      postFCDOccludedCount,
      postFCDValidImageCount,
      postFCDChangeDetectionRepeatability,
      binaryTimeSeriesDecision,
      FCDDecisionMap,
      deltaNDVIChangeDetectionCount,
      fromClassCount,
      toClassCount]
    ),
    changeEvents: changeEvents, // an imagecollection
    fromClassCollection: changeEvents.select("is_from_class"), // an imagecollection
    toClassCollection: changeEvents.select("is_to_class") // an imagecollection
  };

}

exports.run_change_detection = run_change_detection;