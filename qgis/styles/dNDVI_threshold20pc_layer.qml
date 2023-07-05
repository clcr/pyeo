<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis maxScale="0" hasScaleBasedVisibilityFlag="0" version="3.22.3-Białowieża" minScale="1e+08" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal fetchMode="0" mode="0" enabled="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option value="false" type="bool" name="WMSBackgroundLayer"/>
      <Option value="false" type="bool" name="WMSPublishDataSourceUrl"/>
      <Option value="0" type="int" name="embeddedWidgets/count"/>
      <Option value="Value" type="QString" name="identify/format"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option value="" type="QString" name="name"/>
      <Option name="properties"/>
      <Option value="collection" type="QString" name="type"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour" enabled="false"/>
    </provider>
    <rasterrenderer type="singlebandpseudocolor" classificationMin="-1" opacity="1" alphaBand="-1" classificationMax="0" band="1" nodataColor="">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader minimumValue="-1" colorRampType="DISCRETE" classificationMode="2" maximumValue="0" labelPrecision="4" clip="0">
          <colorramp type="gradient" name="[source]">
            <Option type="Map">
              <Option value="247,252,245,255" type="QString" name="color1"/>
              <Option value="30,8,68,255" type="QString" name="color2"/>
              <Option value="0" type="QString" name="discrete"/>
              <Option value="gradient" type="QString" name="rampType"/>
              <Option value="0.2;213,239,207,255:0.4;158,215,152,255:0.6;85,181,103,255:0.8;29,134,65,255" type="QString" name="stops"/>
            </Option>
            <prop v="247,252,245,255" k="color1"/>
            <prop v="30,8,68,255" k="color2"/>
            <prop v="0" k="discrete"/>
            <prop v="gradient" k="rampType"/>
            <prop v="0.2;213,239,207,255:0.4;158,215,152,255:0.6;85,181,103,255:0.8;29,134,65,255" k="stops"/>
          </colorramp>
          <item value="-0.8" color="#f7fcf5" alpha="255" label="&lt;= -0.8000"/>
          <item value="-0.6" color="#c7e9c1" alpha="255" label="-0.8000 - -0.6000"/>
          <item value="-0.4" color="#7ac680" alpha="255" label="-0.6000 - -0.4000"/>
          <item value="-0.2" color="#2b924b" alpha="255" label="-0.4000 - -0.2000"/>
          <item value="inf" color="#1e0844" alpha="255" label="> -0.2000"/>
          <rampLegendSettings maximumLabel="" prefix="" suffix="" direction="0" useContinuousLegend="1" orientation="2" minimumLabel="">
            <numericFormat id="basic">
              <Option type="Map">
                <Option value="" type="QChar" name="decimal_separator"/>
                <Option value="6" type="int" name="decimals"/>
                <Option value="0" type="int" name="rounding_type"/>
                <Option value="false" type="bool" name="show_plus"/>
                <Option value="true" type="bool" name="show_thousand_separator"/>
                <Option value="false" type="bool" name="show_trailing_zeros"/>
                <Option value="" type="QChar" name="thousand_separator"/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0" gamma="1"/>
    <huesaturation colorizeGreen="128" colorizeBlue="128" saturation="0" invertColors="0" colorizeOn="0" grayscaleMode="0" colorizeStrength="100" colorizeRed="255"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
