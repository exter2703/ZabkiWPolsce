"""Kedro pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from zabki_prediction.pipelines.site_selection.nodes import (
    build_model_tables,
    parse_zabka_geojson,
    train_autogluon_and_rank_candidates,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=parse_zabka_geojson,
                inputs="raw_zabka_geojson",
                outputs="store_points",
                name="parse_zabka_geojson",
            ),
            node(
                func=build_model_tables,
                inputs=["store_points", "city_population", "params:modeling"],
                outputs=["training_table", "candidate_table"],
                name="build_model_tables",
            ),
            node(
                func=train_autogluon_and_rank_candidates,
                inputs=["training_table", "candidate_table", "params:modeling", "params:mlflow"],
                outputs=[
                    "model_metrics",
                    "top_candidate_locations",
                    "all_candidate_locations",
                    "city_location_scores",
                    "monitoring_profile",
                ],
                name="train_autogluon_and_rank_candidates",
            ),
        ]
    )
