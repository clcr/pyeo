// needed to allow this file to be required by main.js
const ee = require('@google/earthengine');

/** 
 * This function runs change detection, comparing a classified baseline mosaic to every image in a classified change timeseries. The sequence is intended as:
 * 1. Apply a changeFromClass mask to the classifiedBaseline
 * 2. Apply a changeToClass mask to the classifiedTimeseries
 * 3. Identify the pixels that transition from changeFromClass to changeToClass
 * 4. Compare the NDVI value for each identified pixel against the NDVI in the classifiedBaseline, check if is greater than deltaNDVI
 * 5. Count how many times changes persist, discard if less than minConsecutiveChanges
 * 6. Report the date of the first change per pixel
 * 7. Report the date of the most recent change per pixel
 * 
 * @param {ee.Image} classifiedBaseline - The classified baseline mosaic
 * @param {ee.ImageCollection} classifiedTimeseries - A timeseries of classified change images
 * @param {number} changeFromClass - the class to detect land cover changes from
 * @param {number} changeToClass - the class to detect land cover changes to
 * @param {number} deltaNDVI - the NDVI threshold to filter out changes
 * @param {number} minConsecutiveChanges - the minimum number of consecutive changes for a change to be considered valid
 * @returns {ee.Image} - a multi-band image of the change report, containing pixels of valid change, with the first and last date of these changes
*/
function run_change_detection({
    classifiedBaseline, classifiedTimeseries, changeFromClass, changeToClass, deltaNDVI, minConsecutiveChanges}) {

    // 1: get the baseline and create the initial mask where classes equal the changeFromClass
    const baselineClassification = classifiedBaseline.select("classification");
    const baselineNDVI = classifiedBaseline.select("NDVI");
    const fromMask = baselineClassification.eq(changeFromClass);

    // 2, 3, 4: map over the timeseries to locate valid change events per image
    const changeEvents = classifiedTimeseries.map((image) => {
        const currentClass = image.select("classification");
        const currentNDVI = image.select("NDVI");

        // identify which pixels changed to the changeToClass
        const toMask = currentClass.eq(changeToClass);
        const transitionMask = fromMask.and(toMask);

        // calculate whether the NDVI change is greater than deltaNDVI
        const ndviChange = baselineNDVI.subtract(currentNDVI);
        const ndviMask = ndviChange.gt(deltaNDVI);

        // flag where both conditions (class and NDVI change) are met
        const isChangeMask = transitionMask.and(ndviMask).rename("is_change");

        // store the image date as a band to be accessed later
        const dateMillis = ee.Image.constant(
            image.date().millis())
            .rename("image_date");

        return image.addBands([isChangeMask, dateMillis]);
    });
    
    // 5: track change persistency
    // initialise an empty state image to track streaks and dates over the timeseries
    const initialState = ee.Image([
        ee.Image.constant(0).rename("streak"),
        ee.Image.constant(0).rename("max_streak"),
        ee.Image.constant(0).rename("first_date"),
        ee.Image.constant(0).rename("last_date")
    ]);

    // track the change streaks across the timeseries
    const calculateStreaks = (image, state) => {
        state = ee.Image(state);
        const isChange = image.select("is_change");
        const currentDate = image.select("image_date");

        // create a mask of valid (unclouded) pixels in the image to ensure this does not break a valid streak
        const isValid = isChange.mask();

        // get the current streak, append + 1 if there is a change
        const currentStreak = state.select("streak");
        const newStreak = currentStreak.add(1).multiply(isChange.unmask(0));

        // if pixel is cloud masked, keep the old streak
        // https://developers.google.com/earth-engine/apidocs/ee-image-where
        // "For each pixel in 'currentStreak', if the corresponding pixel in 'isValid' is 1, output the corresponding pixel in newStreak, otherwise output the input pixel."
        const updatedStreak = currentStreak.where(isValid, newStreak);

        const maxStreak = state.select("max_streak");
        const updatedMaxSteak = maxStreak.max(updatedStreak);

        // 6: if streak progresses from 0 to 1 (a change), get image_date
        const firstDate = state.select("image_date");
        const isFirstChange = currentStreak.eq(0).and(updatedStreak.eq(1));
        const updatedFirstDate = firstDate.where(isFirstChange.and(isValid), currentDate);

        // 7: update last_date every time a valid change occurs
        const lastDate = state.select("last_date");
        const updatedLastDate = lastDate.where(isChange.unmask(0).eq(1).and(isValid), currentDate);

        return ee.Image([
            updatedStreak,
            updatedMaxSteak,
            updatedFirstDate,
            updatedLastDate
        ])
    }; // end of calculateStreaks constant

    // run calculateStreaks across the change timeseries
    const finalState = ee.Image(changeEvents.iterate(calculateStreaks, initialState))

    // filter the final output to only include pixels that met the consecutive threshold
    const validPixels = finalState.select("max_streak").gte(minConsecutiveChanges);

    return finalState.select([
        "first_date",
        "last_date"
    ]).updateMask(validPixels);

}; // end of function

module.exports({
    run_change_detection
})