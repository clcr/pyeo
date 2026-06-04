/**
 * builds a cloud-masked Sentinel-2 collection the way
 * SEPAL's Time Series recipe does: one image per acquisition, clouds removed
 * per image with updateMask (no-data, not a fill value), using
 * Cloud Score+ and/or s2cloudless.
*/

exports.build = function (params) {
    params = params || {}
    var aoi = params.aoi
    var startDate = params.startDate
    var endDate = params.endDate
    var method = params.method || 'CLOUD_SCORE_PLUS'
    var maxCloudProbability = params.maxCloudProbability == null ? 65 : params.maxCloudProbability
    var csBand = params.csBand || 'cs_cdf'
    var maxCsProbability = params.maxCsProbability == null ? 45 : params.maxCsProbability
    var useSR = params.useSR || false
    var bands = params.bands || null

    var s2Id = useSR ? 'COPERNICUS/S2_SR_HARMONIZED' : 'COPERNICUS/S2_HARMONIZED'

    var s2 = ee.ImageCollection(s2Id)
        .filterBounds(aoi)
        .filterDate(startDate, endDate)
        .sort("system:time_start", true) // whether to sort ascending

    var s2clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(startDate, endDate)
        .sort("system:time_start", true)

    var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(startDate, endDate)
        .sort("system:time_start", true)


    // linkCollection joins by system:index 
    var linked = s2
        .linkCollection(s2clouds, ['probability'])
        .linkCollection(csPlus, [csBand])

    var maskImage = function (image) {
        var cloudFromProbability = image.select('probability').gt(maxCloudProbability)
        var cloudFromScorePlus = image.select(csBand).lt(1 - maxCsProbability / 100)
        var cloud = ee.Image(
            method === 'S2_CLOUD_PROBABILITY' ? cloudFromProbability :
            method === 'CLOUD_SCORE_PLUS' ? cloudFromScorePlus :
            cloudFromProbability.or(cloudFromScorePlus)   // 'BOTH'
        )
        return image.updateMask(cloud.not())
    }

    var masked = linked.map(maskImage)
    return bands ? masked.select(bands) : masked
}
