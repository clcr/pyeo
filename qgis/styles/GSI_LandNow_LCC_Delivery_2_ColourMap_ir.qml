<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis maxScale="0" styleCategories="AllStyleCategories" minScale="1e+08" hasScaleBasedVisibilityFlag="0" version="3.22.3-Białowieża">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal fetchMode="0" enabled="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option value="false" name="WMSBackgroundLayer" type="bool"/>
      <Option value="false" name="WMSPublishDataSourceUrl" type="bool"/>
      <Option value="0" name="embeddedWidgets/count" type="int"/>
      <Option value="Value" name="identify/format" type="QString"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option value="" name="name" type="QString"/>
      <Option name="properties"/>
      <Option value="collection" name="type" type="QString"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedOutResamplingMethod="nearestNeighbour" zoomedInResamplingMethod="nearestNeighbour" enabled="false" maxOversampling="2"/>
    </provider>
    <rasterrenderer classificationMin="1" classificationMax="11" opacity="1" alphaBand="-1" nodataColor="" band="1" type="singlebandpseudocolor">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader minimumValue="1" classificationMode="1" clip="0" labelPrecision="0" colorRampType="EXACT" maximumValue="11">
          <colorramp name="[source]" type="gradient">
            <Option type="Map">
              <Option value="255,187,34,255" name="color1" type="QString"/>
              <Option value="255,122,65,255" name="color2" type="QString"/>
              <Option value="0" name="discrete" type="QString"/>
              <Option value="gradient" name="rampType" type="QString"/>
              <Option value="0.1;255,255,76,255:0.2;240,150,255,255:0.3;250,0,0,255:0.4;180,180,180,255:0.6;0,50,200,255:0.7;0,150,160,255:0.8;250,230,160,255:0.9;24,135,0,255" name="stops" type="QString"/>
            </Option>
            <prop k="color1" v="255,187,34,255"/>
            <prop k="color2" v="255,122,65,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.1;255,255,76,255:0.2;240,150,255,255:0.3;250,0,0,255:0.4;180,180,180,255:0.6;0,50,200,255:0.7;0,150,160,255:0.8;250,230,160,255:0.9;24,135,0,255"/>
          </colorramp>
          <item color="#ffbb22" label="1, Shrubs (20)" alpha="255" value="1"/>
          <item color="#ffff4c" label="2, Herbaceous Vegetation (30)" alpha="255" value="2"/>
          <item color="#f096ff" label="3, Cropland (40)" alpha="255" value="3"/>
          <item color="#fa0000" label="4,Urban (50)" alpha="255" value="4"/>
          <item color="#b4b4b4" label="5, Bare / sparse vegetation (60)" alpha="255" value="5"/>
          <item color="#0032c8" label="7, Permanent water body (80)" alpha="255" value="7"/>
          <item color="#0096a0" label="8, Herbaceous wetland (90)" alpha="255" value="8"/>
          <item color="#fae6a0" label="9, Moss and Lichen (100)" alpha="255" value="9"/>
          <item color="#188700" label="10, All Forest Classes" alpha="255" value="10"/>
          <item color="#ff7a41" label="11, Coffee" alpha="255" value="11"/>
          <rampLegendSettings useContinuousLegend="1" orientation="2" maximumLabel="" minimumLabel="" prefix="" suffix="" direction="0">
            <numericFormat id="basic">
              <Option type="Map">
                <Option value="" name="decimal_separator" type="QChar"/>
                <Option value="6" name="decimals" type="int"/>
                <Option value="0" name="rounding_type" type="int"/>
                <Option value="false" name="show_plus" type="bool"/>
                <Option value="true" name="show_thousand_separator" type="bool"/>
                <Option value="false" name="show_trailing_zeros" type="bool"/>
                <Option value="" name="thousand_separator" type="QChar"/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0" gamma="1"/>
    <huesaturation grayscaleMode="0" colorizeOn="0" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100" invertColors="0" saturation="0" colorizeRed="255"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
