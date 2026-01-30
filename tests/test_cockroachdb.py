"""
Tests for CockroachDB vector database client.

Assumes CockroachDB is running on localhost:26257.

To start CockroachDB locally:
    cockroach start-single-node --insecure --listen-addr=localhost:26257
"""

import logging

import numpy as np
import pytest

from vectordb_bench.models import DB

log = logging.getLogger(__name__)


def is_cockroachdb_available():
    """Check if CockroachDB is running locally."""
    try:
        import psycopg

        conn = psycopg.connect(
            host="localhost",
            port=26257,
            user="root",
            password="",
            dbname="defaultdb",
            sslmode="disable",
        )
        conn.close()
        return True
    except Exception:
        return False


class TestCockroachDB:
    """Test suite for CockroachDB vector operations."""

    def test_insert_and_search(self):
        """Test basic insert and search operations."""
        assert DB.CockroachDB.value == "CockroachDB"

        dbcls = DB.CockroachDB.init_cls
        dbConfig = DB.CockroachDB.config_cls

        # Connection config (matches your local CockroachDB instance)
        config = {
            "host": "localhost",
            "port": 26257,
            "user_name": "root",
            "password": "",
            "db_name": "defaultdb",
            "table_name": "test_cockroachdb",
        }

        # Note: sslmode=disable is handled in the client's connect_config options

        dim = 128
        count = 1000

        # Initialize CockroachDB client
        cockroachdb = dbcls(
            dim=dim,
            db_config=config,
            db_case_config=None,
            collection_name="test_cockroachdb",
            drop_old=True,
        )

        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        # Test insert
        with cockroachdb.init():
            res = cockroachdb.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))

            assert res[0] == count, f"Insert count mismatch: {res[0]} != {count}"
            assert res[1] is None, f"Insert failed with error: {res[1]}"

        # Test search
        with cockroachdb.init():
            test_id = np.random.randint(count)
            q = embeddings[test_id]

            res = cockroachdb.search_embedding(query=q, k=10)

            assert len(res) > 0, "Search returned no results"
            assert res[0] == int(test_id), f"Top result {res[0]} != query id {test_id}"

        log.info("CockroachDB insert and search test passed")

    def test_search_with_filter(self):
        """Test search with filters."""
        assert DB.CockroachDB.value == "CockroachDB"

        dbcls = DB.CockroachDB.init_cls

        config = {
            "host": "localhost",
            "port": 26257,
            "user_name": "root",
            "password": "",
            "db_name": "defaultdb",
            "table_name": "test_cockroachdb_filter",
        }

        dim = 128
        count = 1000
        filter_value = 0.9

        cockroachdb = dbcls(
            dim=dim,
            db_config=config,
            db_case_config=None,
            collection_name="test_cockroachdb_filter",
            drop_old=True,
        )

        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        # Insert data
        with cockroachdb.init():
            res = cockroachdb.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
            assert res[0] == count, "Insert count mismatch"

        # Search with filter
        with cockroachdb.init():
            filter_id = int(count * filter_value)
            test_id = np.random.randint(filter_id, count)
            q = embeddings[test_id]

            from vectordb_bench.backend.filter import IntFilter

            filters = IntFilter(int_value=filter_id, filter_rate=0.9)
            cockroachdb.prepare_filter(filters)

            res = cockroachdb.search_embedding(query=q, k=10)

            assert len(res) > 0, "Filtered search returned no results"
            assert res[0] == int(test_id), f"Top result {res[0]} != query id {test_id}"

            # Verify all results are >= filter_value
            for result_id in res:
                assert int(result_id) >= filter_id, f"Result {result_id} < filter threshold {filter_id}"

        log.info("CockroachDB filter test passed")

    @pytest.mark.skipif(not is_cockroachdb_available(), reason="CockroachDB not available")
    def test_metadata_index_creation(self):
        """Test metadata index creation for optimized filtered searches."""
        from vectordb_bench.backend.clients.api import MetricType
        from vectordb_bench.backend.clients.cockroachdb.config import CockroachDBVectorIndexConfig

        dbcls = DB.CockroachDB.init_cls

        config = {
            "host": "localhost",
            "port": 26257,
            "user_name": "root",
            "password": "",
            "db_name": "defaultdb",
            "table_name": "test_metadata_idx",
        }

        case_config = CockroachDBVectorIndexConfig(
            metric_type=MetricType.L2,
            create_index_after_load=True,
            create_metadata_index=True,
        )

        dim = 128
        count = 100

        cockroachdb = dbcls(
            dim=dim,
            db_config=config,
            db_case_config=case_config,
            collection_name="test_metadata_idx",
            drop_old=True,
        )

        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        with cockroachdb.init():
            res = cockroachdb.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
            assert res[0] == count

        cockroachdb.optimize()

        # Verify metadata index was created
        import psycopg

        conn = psycopg.connect(**cockroachdb.connect_config)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = %s",
            ("test_metadata_idx",),
        )
        indexes = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        assert "test_metadata_idx_metadata_idx" in indexes, f"Metadata index not found. Indexes: {indexes}"
        log.info("CockroachDB metadata index test passed")

    @pytest.mark.skipif(not is_cockroachdb_available(), reason="CockroachDB not available")
    def test_search_with_metadata_index(self):
        """Test that filtered search works with metadata index."""
        from vectordb_bench.backend.clients.api import MetricType
        from vectordb_bench.backend.clients.cockroachdb.config import CockroachDBVectorIndexConfig
        from vectordb_bench.backend.filter import IntFilter

        dbcls = DB.CockroachDB.init_cls

        config = {
            "host": "localhost",
            "port": 26257,
            "user_name": "root",
            "password": "",
            "db_name": "defaultdb",
            "table_name": "test_search_metadata_idx",
        }

        case_config = CockroachDBVectorIndexConfig(
            metric_type=MetricType.L2,
            create_index_after_load=True,
            create_metadata_index=True,
        )

        dim = 128
        count = 1000

        cockroachdb = dbcls(
            dim=dim,
            db_config=config,
            db_case_config=case_config,
            collection_name="test_search_metadata_idx",
            drop_old=True,
        )

        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        with cockroachdb.init():
            res = cockroachdb.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
            assert res[0] == count

        cockroachdb.optimize()

        # Search with filter using metadata index
        with cockroachdb.init():
            filter_threshold = 900
            test_id = np.random.randint(filter_threshold, count)
            q = embeddings[test_id]

            filters = IntFilter(int_value=filter_threshold, filter_rate=0.9)
            cockroachdb.prepare_filter(filters)

            res = cockroachdb.search_embedding(query=q, k=10)

            assert len(res) > 0, "Filtered search returned no results"
            for result_id in res:
                assert int(result_id) >= filter_threshold

        log.info("CockroachDB search with metadata index test passed")

    @pytest.mark.skipif(not is_cockroachdb_available(), reason="CockroachDB not available")
    def test_create_index_before_load(self):
        """Test creating vector index before loading data (faster for large datasets)."""
        from vectordb_bench.backend.clients.api import MetricType
        from vectordb_bench.backend.clients.cockroachdb.config import CockroachDBVectorIndexConfig

        dbcls = DB.CockroachDB.init_cls

        config = {
            "host": "localhost",
            "port": 26257,
            "user_name": "root",
            "password": "",
            "db_name": "defaultdb",
            "table_name": "test_index_before_load",
        }

        case_config = CockroachDBVectorIndexConfig(
            metric_type=MetricType.L2,
            create_index_before_load=True,
            create_index_after_load=False,
        )

        dim = 128
        count = 100

        cockroachdb = dbcls(
            dim=dim,
            db_config=config,
            db_case_config=case_config,
            collection_name="test_index_before_load",
            drop_old=True,
        )

        # Verify index exists before load
        import psycopg

        conn = psycopg.connect(**cockroachdb.connect_config)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT index_name FROM [SHOW INDEXES FROM test_index_before_load] "
            f"WHERE index_name = 'test_index_before_load_vector_idx'"
        )
        indexes_before = cursor.fetchall()
        cursor.close()
        conn.close()

        assert len(indexes_before) > 0, "Vector index should exist before loading data"

        # Insert data with existing index (tests retry logic for RETRY_SERIALIZABLE)
        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        with cockroachdb.init():
            res = cockroachdb.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
            assert res[0] == count, f"Insert with pre-existing index failed: {res}"

        # Test search works
        with cockroachdb.init():
            test_id = 0
            q = embeddings[test_id]
            res = cockroachdb.search_embedding(query=q, k=10)
            assert len(res) > 0, "Search with pre-created index returned no results"

        log.info("CockroachDB create_index_before_load test passed")

    @pytest.mark.skipif(not is_cockroachdb_available(), reason="CockroachDB not available")
    def test_index_creation_timeout_configuration(self):
        """Test that index_creation_timeout is configurable."""
        from vectordb_bench.backend.clients.api import MetricType
        from vectordb_bench.backend.clients.cockroachdb.config import CockroachDBVectorIndexConfig

        # Test custom timeout value
        case_config = CockroachDBVectorIndexConfig(
            metric_type=MetricType.COSINE,
            create_index_after_load=True,
            index_creation_timeout=1200,  # 20 minutes
        )

        assert case_config.index_creation_timeout == 1200, "Custom timeout not set correctly"

        # Test default timeout value
        case_config_default = CockroachDBVectorIndexConfig(
            metric_type=MetricType.COSINE,
            create_index_after_load=True,
        )

        assert case_config_default.index_creation_timeout == 1200, "Default timeout should be 1200 seconds"
        log.info("CockroachDB index_creation_timeout configuration test passed")

    @pytest.mark.skipif(not is_cockroachdb_available(), reason="CockroachDB not available")
    def test_retry_logic_with_rollback(self):
        """Test that insert operations handle errors with proper rollback."""
        from vectordb_bench.backend.clients.api import MetricType
        from vectordb_bench.backend.clients.cockroachdb.config import CockroachDBVectorIndexConfig

        dbcls = DB.CockroachDB.init_cls

        config = {
            "host": "localhost",
            "port": 26257,
            "user_name": "root",
            "password": "",
            "db_name": "defaultdb",
            "table_name": "test_retry_rollback",
        }

        case_config = CockroachDBVectorIndexConfig(
            metric_type=MetricType.L2,
            create_index_before_load=True,  # This increases chance of serialization conflicts
            create_index_after_load=False,
        )

        dim = 64
        count = 50

        cockroachdb = dbcls(
            dim=dim,
            db_config=config,
            db_case_config=case_config,
            collection_name="test_retry_rollback",
            drop_old=True,
        )

        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        # Insert should succeed even if there are transient errors (retry handles them)
        with cockroachdb.init():
            res = cockroachdb.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
            assert res[0] == count, f"Insert with retry logic failed: {res}"
            assert res[1] is None, f"Insert returned error: {res[1]}"

        # Verify data was actually inserted
        import psycopg

        conn = psycopg.connect(**cockroachdb.connect_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_retry_rollback")
        actual_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        assert actual_count == count, f"Actual count {actual_count} != expected {count}"
        log.info("CockroachDB retry logic with rollback test passed")
