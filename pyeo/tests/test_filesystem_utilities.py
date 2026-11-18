"""
Unit tests for pyeo.filesystem_utilities module.

These tests cover basic utility functions for timestamp parsing,
file naming conventions, and path manipulation specific to Sentinel-2 data.
"""

import pytest
import os
from datetime import datetime


# Note: We'll test functions that don't require actual satellite data files
# These are the simplest functions to test and provide good coverage


class TestSentinel2TimestampParsing:
"""Tests for Sentinel-2 timestamp parsing functions."""

def test_parse_valid_sentinel2_timestamp(self):
"""Test parsing a valid Sentinel-2 timestamp format."""
# Sentinel-2 uses format: yyyymmddThhmmss
# Example: 20231215T103045 = Dec 15, 2023 at 10:30:45
from pyeo.filesystem_utilities import get_sen_2_image_timestamp

# Test with a typical S2 filename
test_filename = "S2A_MSIL2A_20231215T103045_N0509_R108_T36MYE_20231215T134521.SAFE"

timestamp = get_sen_2_image_timestamp(test_filename)

# Timestamp should be extracted as: 20231215T103045
assert timestamp == "20231215T103045"

def test_parse_timestamp_from_different_formats(self):
"""Test timestamp parsing from various S2 filename formats."""
from pyeo.filesystem_utilities import get_sen_2_image_timestamp

# Test L1C format
l1c_name = "S2B_MSIL1C_20220601T073619_N0400_R092_T36MYE_20220601T094032.SAFE"
assert get_sen_2_image_timestamp(l1c_name) == "20220601T073619"

# Test L2A format
l2a_name = "S2A_MSIL2A_20220601T073619_N0400_R092_T36MYE_20220601T094032.SAFE"
assert get_sen_2_image_timestamp(l2a_name) == "20220601T073619"


class TestSentinel2TileExtraction:
"""Tests for extracting Sentinel-2 tile information from filenames."""

def test_extract_tile_from_filename(self):
"""Test extraction of tile identifier from S2 filename."""
from pyeo.filesystem_utilities import get_sen_2_image_tile

test_filename = "S2A_MSIL2A_20231215T103045_N0509_R108_T36MYE_20231215T134521.SAFE"

tile = get_sen_2_image_tile(test_filename)

# Should extract: 36MYE
assert tile == "36MYE"

def test_extract_different_tiles(self):
"""Test tile extraction from various geographic locations."""
from pyeo.filesystem_utilities import get_sen_2_image_tile

# Kenya tile
kenya_file = "S2A_MSIL2A_20220101T073619_N0400_R092_T36MZE_20220101T094032.SAFE"
assert get_sen_2_image_tile(kenya_file) == "36MZE"

# UK tile
uk_file = "S2B_MSIL2A_20220601T113339_N0400_R080_T30UXC_20220601T143521.SAFE"
assert get_sen_2_image_tile(uk_file) == "30UXC"


class TestSerialDateConversion:
"""Tests for serial date to string conversion."""

def test_serial_date_to_string_basic(self):
"""Test conversion of serial date (days since epoch) to string."""
from pyeo.filesystem_utilities import serial_date_to_string

# Day 0 should be January 1, 2000
result = serial_date_to_string(0)
assert result == "20000101T000000"

def test_serial_date_one_year_later(self):
"""Test serial date one year after epoch."""
from pyeo.filesystem_utilities import serial_date_to_string

# 365 days after Jan 1, 2000 = Dec 31, 2000
result = serial_date_to_string(365)
assert result == "20001231T000000"

def test_serial_date_recent(self):
"""Test serial date for a recent date."""
from pyeo.filesystem_utilities import serial_date_to_string

# Approximately 8766 days = ~24 years = around 2024
# This is an approximate test
result = serial_date_to_string(8766)
# Should be in 2024
assert result.startswith("2024")


class TestFileSorting:
"""Tests for sorting files by timestamp."""

def test_sort_files_by_timestamp(self):
"""Test that files are correctly sorted chronologically."""
from pyeo.filesystem_utilities import sort_by_timestamp

# Create test filenames with different timestamps
files = [
"S2A_MSIL2A_20231215T103045_N0509_R108_T36MYE_20231215T134521.SAFE",
"S2A_MSIL2A_20230101T103045_N0509_R108_T36MYE_20230101T134521.SAFE",
"S2A_MSIL2A_20231130T103045_N0509_R108_T36MYE_20231130T134521.SAFE",
]

sorted_files = sort_by_timestamp(files)

# Should be sorted: Jan 1 -> Nov 30 -> Dec 15
assert "20230101" in sorted_files[0]
assert "20231130" in sorted_files[1]
assert "20231215" in sorted_files[2]


class TestGranuleIDExtraction:
"""Tests for Sentinel-2 granule ID extraction."""

def test_extract_granule_id(self):
"""Test extraction of complete granule ID from filename."""
from pyeo.filesystem_utilities import get_sen_2_granule_id

test_filename = "S2A_MSIL2A_20231215T103045_N0509_R108_T36MYE_20231215T134521.SAFE"

granule_id = get_sen_2_granule_id(test_filename)

# Granule ID should include satellite, timestamp, and tile
assert "S2A" in granule_id
assert "20231215T103045" in granule_id
assert "36MYE" in granule_id


class TestProductTypeDetection:
"""Tests for detecting Sentinel-2 product types (L1C vs L2A)."""

def test_detect_l1c_product(self):
"""Test detection of L1C (Top of Atmosphere) product."""
from pyeo.filesystem_utilities import get_safe_product_type

l1c_file = "S2A_MSIL1C_20231215T103045_N0509_R108_T36MYE_20231215T134521.SAFE"

product_type = get_safe_product_type(l1c_file)

assert product_type == "MSIL1C"

def test_detect_l2a_product(self):
"""Test detection of L2A (Bottom of Atmosphere) product."""
from pyeo.filesystem_utilities import get_safe_product_type

l2a_file = "S2A_MSIL2A_20231215T103045_N0509_R108_T36MYE_20231215T134521.SAFE"

product_type = get_safe_product_type(l2a_file)

assert product_type == "MSIL2A"


class TestBaselineExtraction:
"""Tests for extracting processing baseline from filenames."""

def test_extract_baseline_number(self):
"""Test extraction of processing baseline version."""
from pyeo.filesystem_utilities import get_sen_2_baseline

# N0509 means baseline version 05.09
test_filename = "S2A_MSIL2A_20231215T103045_N0509_R108_T36MYE_20231215T134521.SAFE"

baseline = get_sen_2_baseline(test_filename)

# Should extract: 0509 or similar
assert "05" in baseline or "N0509" in baseline


# Test configuration and fixtures
@pytest.fixture
def sample_sentinel2_filenames():
"""Fixture providing a list of sample Sentinel-2 filenames for testing."""
return [
"S2A_MSIL2A_20231215T103045_N0509_R108_T36MYE_20231215T134521.SAFE",
"S2B_MSIL1C_20220601T073619_N0400_R092_T36MZE_20220601T094032.SAFE",
"S2A_MSIL2A_20230101T103045_N0509_R108_T30UXC_20230101T134521.SAFE",
]


@pytest.fixture
def temp_directory(tmp_path):
"""Fixture providing a temporary directory for file operations."""
return tmp_path


def test_fixtures_work(sample_sentinel2_filenames):
"""Meta-test to verify fixtures are working."""
assert len(sample_sentinel2_filenames) == 3
assert all(".SAFE" in f for f in sample_sentinel2_filenames)


# Run tests with: pytest test_filesystem_utilities.py -v
# For coverage: pytest test_filesystem_utilities.py --cov=pyeo.filesystem_utilities --cov-report=html