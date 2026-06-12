import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
from numcodecs import Zstd

from opal_studio.image_loader import open_spatialdata_collection


class SpatialDataLoaderTest(unittest.TestCase):
    def test_collection_uses_consolidated_metadata_and_multichannel_chunks(self):
        group_meta = {
            "attributes": {
                "ome": {
                    "omero": {"channels": [{"label": 0}, {"label": 1}]},
                    "multiscales": [{
                        "axes": [
                            {"name": "c", "type": "channel"},
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [{
                            "path": "0",
                            "coordinateTransformations": [{
                                "type": "scale",
                                "scale": [1.0, 1.0, 1.0],
                            }],
                        }],
                    }],
                }
            },
            "zarr_format": 3,
            "node_type": "group",
        }
        array_meta = {
            "shape": [2, 3, 4],
            "data_type": "int16",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [2, 3, 4]},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
            ],
            "attributes": {},
            "zarr_format": 3,
            "node_type": "array",
            "storage_transformers": [],
        }
        root_meta = {
            "attributes": {"spatialdata_attrs": {"version": "0.2"}},
            "zarr_format": 3,
            "consolidated_metadata": {
                "kind": "inline",
                "must_understand": False,
                "metadata": {
                    "images": {
                        "attributes": {},
                        "zarr_format": 3,
                        "node_type": "group",
                    },
                    "images/s001": group_meta,
                    "images/s001/0": array_meta,
                },
            },
            "node_type": "group",
        }

        metadata_file = mock_open(read_data=json.dumps(root_meta))
        with patch.object(Path, "is_dir", return_value=True), \
                patch.object(Path, "exists", return_value=True), \
                patch("opal_studio.image_loader._parse_mcd_panel", return_value={}), \
                patch("builtins.open", metadata_file):
            collection = open_spatialdata_collection("sections.zarr")
            image = collection.open_image(0)

        self.assertEqual(collection.image_names, ["s001"])
        self.assertEqual(image.channel_names, ["0", "1"])

        data = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
        chunk_file = mock_open(read_data=Zstd().encode(data.tobytes()))
        with patch.object(Path, "exists", return_value=True), \
                patch("builtins.open", chunk_file):
            np.testing.assert_array_equal(
                image.levels[0]._zarr[1, 0:3, 0:4],
                data[1],
            )
            self.assertEqual(image.levels[0]._zarr.channel_maxima(), [11.0, 23.0])


if __name__ == "__main__":
    unittest.main()
