"""CLI parameter definitions for CockroachDB."""

from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class CockroachDBTypedDict(CommonTypedDict):
    """Type definition for CockroachDB CLI parameters."""

    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="CockroachDB username", default="root", show_default=True),
    ]
    password: Annotated[
        str,
        click.option("--password", type=str, help="CockroachDB password", default="", show_default=False),
    ]
    host: Annotated[
        str,
        click.option("--host", type=str, help="CockroachDB host", required=True),
    ]
    port: Annotated[
        int,
        click.option("--port", type=int, help="CockroachDB port", default=26257, show_default=True),
    ]
    db_name: Annotated[
        str,
        click.option("--db-name", type=str, help="Database name", required=True),
    ]
    sslmode: Annotated[
        str,
        click.option(
            "--sslmode",
            type=str,
            help="SSL mode (disable, require, verify-ca, verify-full)",
            default="disable",
            show_default=True,
        ),
    ]
    sslrootcert: Annotated[
        str | None,
        click.option(
            "--sslrootcert",
            type=str,
            help="Path to SSL root certificate (required for verify-ca, verify-full)",
            default=None,
        ),
    ]
    min_partition_size: Annotated[
        int | None,
        click.option(
            "--min-partition-size",
            type=int,
            help="Minimum vectors per partition (default: 16, range: 1-1024)",
            default=16,
            show_default=True,
        ),
    ]
    max_partition_size: Annotated[
        int | None,
        click.option(
            "--max-partition-size",
            type=int,
            help="Maximum vectors per partition (default: 128, range: 4x min-4096)",
            default=128,
            show_default=True,
        ),
    ]
    vector_search_beam_size: Annotated[
        int | None,
        click.option(
            "--vector-search-beam-size",
            type=int,
            help="Partitions explored during search (default: 32)",
            default=32,
            show_default=True,
        ),
    ]
    create_metadata_index: Annotated[
        bool,
        click.option(
            "--create-metadata-index",
            is_flag=True,
            help="Create B-tree index on metadata_id for faster filtered searches",
            default=False,
            show_default=True,
        ),
    ]
    create_label_vector_index: Annotated[
        bool,
        click.option(
            "--create-label-vector-index",
            is_flag=True,
            help="Create composite vector index on (label, embedding) for label filtering",
            default=False,
            show_default=True,
        ),
    ]
    create_index_before_load: Annotated[
        bool,
        click.option(
            "--create-index-before-load",
            is_flag=True,
            help="Create vector index before loading data (faster for large datasets)",
            default=False,
            show_default=True,
        ),
    ]
    index_creation_timeout: Annotated[
        int,
        click.option(
            "--index-creation-timeout",
            type=int,
            help="Timeout in seconds for index creation (default: 1200)",
            default=1200,
            show_default=True,
        ),
    ]
    insert_max_retries: Annotated[
        int,
        click.option(
            "--insert-max-retries",
            type=int,
            help="Maximum retry attempts for insert operations (default: 5)",
            default=5,
            show_default=True,
        ),
    ]
    insert_retry_initial_delay: Annotated[
        float,
        click.option(
            "--insert-retry-initial-delay",
            type=float,
            help="Initial delay in seconds before first retry (default: 0.5)",
            default=0.5,
            show_default=True,
        ),
    ]
    insert_retry_backoff_factor: Annotated[
        float,
        click.option(
            "--insert-retry-backoff-factor",
            type=float,
            help="Exponential backoff multiplier for retries (default: 2.0)",
            default=2.0,
            show_default=True,
        ),
    ]
    pool_max_idle: Annotated[
        int,
        click.option(
            "--pool-max-idle",
            type=int,
            help="Maximum idle time in seconds before closing pool connections (default: 300)",
            default=300,
            show_default=True,
        ),
    ]
    pool_reconnect_timeout: Annotated[
        float,
        click.option(
            "--pool-reconnect-timeout",
            type=float,
            help="Timeout in seconds for pool reconnection attempts (default: 10.0)",
            default=10.0,
            show_default=True,
        ),
    ]
    statement_timeout: Annotated[
        int,
        click.option(
            "--statement-timeout",
            type=int,
            help="Query statement timeout in seconds (default: 60, used for search queries)",
            default=60,
            show_default=True,
        ),
    ]
    index_poll_interval: Annotated[
        int,
        click.option(
            "--index-poll-interval",
            type=int,
            help="Interval in seconds between index status checks (default: 5)",
            default=5,
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(CockroachDBTypedDict)
def CockroachDB(
    **parameters: Unpack[CockroachDBTypedDict],
):
    """Run CockroachDB vector benchmark."""
    from .config import CockroachDBConfig, CockroachDBVectorIndexConfig

    parameters["custom_case"] = get_custom_case_config(parameters)

    from vectordb_bench.backend.clients.api import MetricType

    # Use provided metric_type or default to COSINE
    metric_type = parameters.get("metric_type")
    if metric_type is None:
        metric_type = MetricType.COSINE
    elif isinstance(metric_type, str):
        metric_type = MetricType(metric_type)

    run(
        db=DB.CockroachDB,
        db_config=CockroachDBConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            sslmode=parameters.get("sslmode", "disable"),
            sslrootcert=parameters.get("sslrootcert"),
            pool_max_idle=parameters.get("pool_max_idle", 300),
            pool_reconnect_timeout=parameters.get("pool_reconnect_timeout", 10.0),
            statement_timeout=parameters.get("statement_timeout", 60),
        ),
        db_case_config=CockroachDBVectorIndexConfig(
            metric_type=metric_type,
            min_partition_size=parameters.get("min_partition_size", 16),
            max_partition_size=parameters.get("max_partition_size", 128),
            vector_search_beam_size=parameters.get("vector_search_beam_size", 32),
            create_metadata_index=parameters.get("create_metadata_index", False),
            create_label_vector_index=parameters.get("create_label_vector_index", False),
            create_index_before_load=parameters.get("create_index_before_load", False),
            create_index_after_load=not parameters.get("create_index_before_load", False),
            index_creation_timeout=parameters.get("index_creation_timeout", 1200),
            index_poll_interval=parameters.get("index_poll_interval", 5),
            insert_max_retries=parameters.get("insert_max_retries", 5),
            insert_retry_initial_delay=parameters.get("insert_retry_initial_delay", 0.5),
            insert_retry_backoff_factor=parameters.get("insert_retry_backoff_factor", 2.0),
        ),
        **parameters,
    )
