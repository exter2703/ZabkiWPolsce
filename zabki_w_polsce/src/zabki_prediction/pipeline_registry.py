"""Pipeline registry for Kedro."""

from kedro.pipeline import Pipeline

from zabki_prediction.pipelines.site_selection.pipeline import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    site_selection = create_pipeline()
    return {
        "__default__": site_selection,
        "site_selection": site_selection,
    }
