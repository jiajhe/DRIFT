"""Data APIs with optional heavy dependencies loaded only on first use."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "ConceptAsset": "concepts",
    "attach_concept_rows": "concepts",
    "concept_route_pad_cross_sections": "concepts",
    "load_concept_asset": "concepts",
    "Episode": "episodes",
    "build_calendar_month_episodes": "episodes",
    "build_cross_split_calendar_month_episodes": "episodes",
    "build_episodes": "episodes",
    "label_availability_map": "episodes",
    "CrossSectionalDays": "qlib_handler",
    "build_handler_bundle": "qlib_handler",
    "pad_cross_sections": "qlib_handler",
    "FullEpochIterator": "temporal_cross_section",
    "SplitCoverage": "temporal_cross_section",
    "TemporalChunk": "temporal_cross_section",
    "TemporalDay": "temporal_cross_section",
    "TemporalQlibSource": "temporal_cross_section",
    "TemporalSplit": "temporal_cross_section",
    "build_temporal_qlib_source": "temporal_cross_section",
    "configure_temporal_sampler": "temporal_cross_section",
    "exact_daily_positions": "temporal_cross_section",
    "materialize_temporal_split": "temporal_cross_section",
}

__all__ = [
    "ConceptAsset",
    "CrossSectionalDays",
    "Episode",
    "FullEpochIterator",
    "SplitCoverage",
    "TemporalChunk",
    "TemporalDay",
    "TemporalQlibSource",
    "TemporalSplit",
    "attach_concept_rows",
    "build_calendar_month_episodes",
    "build_cross_split_calendar_month_episodes",
    "build_episodes",
    "build_handler_bundle",
    "build_temporal_qlib_source",
    "configure_temporal_sampler",
    "concept_route_pad_cross_sections",
    "exact_daily_positions",
    "label_availability_map",
    "load_concept_asset",
    "materialize_temporal_split",
    "pad_cross_sections",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
