<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.22.3-Białowieża" hasScaleBasedVisibilityFlag="0" maxScale="0" minScale="1e+08">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" fetchMode="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option name="WMSBackgroundLayer" type="bool" value="false"/>
      <Option name="WMSPublishDataSourceUrl" type="bool" value="false"/>
      <Option name="embeddedWidgets/count" type="int" value="0"/>
      <Option name="identify/format" type="QString" value="Value"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option name="name" type="QString" value=""/>
      <Option name="properties"/>
      <Option name="type" type="QString" value="collection"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling maxOversampling="2" enabled="false" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer band="1" nodataColor="" alphaBand="-1" classificationMax="1" classificationMin="-1" type="singlebandpseudocolor" opacity="1">
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
        <colorrampshader minimumValue="-1" colorRampType="INTERPOLATED" labelPrecision="4" maximumValue="1" clip="0" classificationMode="1">
          <colorramp name="[source]" type="gradient">
            <Option type="Map">
              <Option name="color1" type="QString" value="247,252,245,255"/>
              <Option name="color2" type="QString" value="0,68,27,255"/>
              <Option name="discrete" type="QString" value="0"/>
              <Option name="rampType" type="QString" value="gradient"/>
              <Option name="stops" type="QString" value="0.13;229,245,224,255:0.26;199,233,192,255:0.39;161,217,155,255:0.52;116,196,118,255:0.65;65,171,93,255:0.78;35,139,69,255:0.9;0,109,44,255"/>
            </Option>
            <prop k="color1" v="247,252,245,255"/>
            <prop k="color2" v="0,68,27,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.13;229,245,224,255:0.26;199,233,192,255:0.39;161,217,155,255:0.52;116,196,118,255:0.65;65,171,93,255:0.78;35,139,69,255:0.9;0,109,44,255"/>
          </colorramp>
          <item label="-1.0000" color="#f7fcf5" alpha="255" value="-1"/>
          <item label="-0.7400" color="#e5f5e0" alpha="255" value="-0.74"/>
          <item label="-0.4800" color="#c7e9c0" alpha="255" value="-0.48"/>
          <item label="-0.2200" color="#a1d99b" alpha="255" value="-0.22"/>
          <item label="0.0400" color="#74c476" alpha="255" value="0.04"/>
          <item label="0.3000" color="#41ab5d" alpha="255" value="0.3"/>
          <item label="0.5600" color="#238b45" alpha="255" value="0.56"/>
          <item label="0.8000" color="#006d2c" alpha="255" value="0.8"/>
          <item label="1.0000" color="#00441b" alpha="255" value="1"/>
          <rampLegendSettings suffix="" useContinuousLegend="1" direction="0" minimumLabel="" maximumLabel="" prefix="" orientation="2">
            <numericFormat id="basic">
              <Option type="Map">
                <Option name="decimal_separator" type="QChar" value=""/>
                <Option name="decimals" type="int" value="6"/>
                <Option name="rounding_type" type="int" value="0"/>
                <Option name="show_plus" type="bool" value="false"/>
                <Option name="show_thousand_separator" type="bool" value="true"/>
                <Option name="show_trailing_zeros" type="bool" value="false"/>
                <Option name="thousand_separator" type="QChar" value=""/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast gamma="1" brightness="0" contrast="0"/>
    <huesaturation colorizeOn="0" saturation="0" colorizeGreen="128" invertColors="0" colorizeRed="255" colorizeStrength="100" colorizeBlue="128" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
