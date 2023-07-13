<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.3-Białowieża" minScale="1e+08" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" maxScale="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" mode="0" fetchMode="0">
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
      <resampling zoomedInResamplingMethod="nearestNeighbour" enabled="false" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer nodataColor="" classificationMin="0" opacity="1" band="5" classificationMax="100" alphaBand="-1" type="singlebandpseudocolor">
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
        <colorrampshader clip="0" minimumValue="0" classificationMode="1" labelPrecision="0" maximumValue="100" colorRampType="INTERPOLATED">
          <colorramp name="[source]" type="gradient">
            <Option type="Map">
              <Option name="color1" type="QString" value="247,252,245,255"/>
              <Option name="color2" type="QString" value="0,68,27,255"/>
              <Option name="discrete" type="QString" value="0"/>
              <Option name="rampType" type="QString" value="gradient"/>
              <Option name="stops" type="QString" value="0.13;229,245,224,255:0.26;199,233,192,255:0.39;161,217,155,255:0.52;116,196,118,255:0.65;65,171,93,255:0.78;35,139,69,255:0.9;0,109,44,255"/>
            </Option>
            <prop v="247,252,245,255" k="color1"/>
            <prop v="0,68,27,255" k="color2"/>
            <prop v="0" k="discrete"/>
            <prop v="gradient" k="rampType"/>
            <prop v="0.13;229,245,224,255:0.26;199,233,192,255:0.39;161,217,155,255:0.52;116,196,118,255:0.65;65,171,93,255:0.78;35,139,69,255:0.9;0,109,44,255" k="stops"/>
          </colorramp>
          <item color="#f7fcf5" alpha="255" label="0" value="0"/>
          <item color="#e5f5e0" alpha="255" label="13" value="13"/>
          <item color="#c7e9c0" alpha="255" label="26" value="26"/>
          <item color="#a1d99b" alpha="255" label="39" value="39"/>
          <item color="#74c476" alpha="255" label="52" value="52"/>
          <item color="#41ab5d" alpha="255" label="65" value="65"/>
          <item color="#238b45" alpha="255" label="78" value="78"/>
          <item color="#006d2c" alpha="255" label="90" value="90"/>
          <item color="#00441b" alpha="255" label="100" value="100"/>
          <rampLegendSettings useContinuousLegend="1" maximumLabel="" direction="0" minimumLabel="" prefix="" suffix="" orientation="2">
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
    <brightnesscontrast contrast="0" brightness="0" gamma="1"/>
    <huesaturation saturation="0" invertColors="0" colorizeRed="255" grayscaleMode="0" colorizeBlue="128" colorizeStrength="100" colorizeGreen="128" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
