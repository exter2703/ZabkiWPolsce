from zabki_prediction.pipelines.site_selection.nodes import parse_zabka_geojson


def test_parse_zabka_geojson_keeps_point_features():
    raw = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "node/1",
                "properties": {"addr:city": "Poznan", "addr:street": "Testowa"},
                "geometry": {"type": "Point", "coordinates": [16.9, 52.4]},
            },
            {
                "type": "Feature",
                "id": "way/2",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": []},
            },
        ],
    }

    frame = parse_zabka_geojson(raw)

    assert len(frame) == 1
    assert frame.loc[0, "store_id"] == "node/1"
    assert frame.loc[0, "city"] == "Poznan"
