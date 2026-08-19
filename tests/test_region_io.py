import unittest

from opal_studio.region_io import geojson_region_parts


class GeoJsonRegionPartsTest(unittest.TestCase):
    def test_polygon_keeps_its_name_without_a_number(self):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [2, 0], [0, 2], [0, 0]]],
            },
            "properties": {"name": "Lung"},
        }

        parts = geojson_region_parts(feature, "Region 1")

        self.assertEqual([name for name, _ in parts], ["Lung"])

    def test_multipolygon_returns_and_numbers_every_polygon(self):
        first_ring = [[0, 0], [2, 0], [0, 2], [0, 0]]
        second_ring = [[10, 10], [12, 10], [10, 12], [10, 10]]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[first_ring], [second_ring]],
            },
            "properties": {"name": "Lesion"},
        }

        parts = geojson_region_parts(feature, "Region 1")

        self.assertEqual([name for name, _ in parts], ["Lesion 1", "Lesion 2"])
        self.assertEqual([ring for _, ring in parts], [first_ring, second_ring])


if __name__ == "__main__":
    unittest.main()
