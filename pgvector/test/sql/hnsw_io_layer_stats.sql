-- ============================================================
-- HNSW Per-Layer Read Statistics -- Regression Tests
-- ============================================================

-- Setup
SET hnsw.io_layer_track = on;
SELECT setseed(0.5);
DROP TABLE IF EXISTS test_layer_stats CASCADE;
CREATE TABLE test_layer_stats (id serial, embedding vector(128));

-- Test 1: Reset clears all counters + GUC visibility (L2)
SHOW hnsw.io_layer_track;  -- Expected: on

SELECT hnsw_io_stats_reset();
SELECT count(*) AS empty_layers FROM hnsw_io_layer_stats();
-- Expected: 0

-- Test 2: INSERT triggers per-layer statistics
CREATE INDEX ON test_layer_stats USING hnsw (embedding vector_l2_ops) WITH (m = 16);

INSERT INTO test_layer_stats (embedding)
SELECT (SELECT array_agg(random()::real)::vector FROM generate_series(1,128))
FROM generate_series(1,100);

-- Verify: at least one layer has non-zero counts
SELECT count(*) > 0 AS has_layer_data FROM hnsw_io_layer_stats();

-- Test 3: Invariant -- io_read_pages <= buffer_read_pages for every row
SELECT count(*) = 0 AS io_le_buffer_invariant
FROM hnsw_io_layer_stats()
WHERE io_read_pages > buffer_read_pages;

-- Test 4: Invariant -- sum of per-layer buffer reads <= global read_pages
SELECT
  (SELECT coalesce(sum(buffer_read_pages), 0) FROM hnsw_io_layer_stats()) <= read_pages
  AS layer_le_global
FROM hnsw_io_stats();

-- Test 5: SCAN/query triggers per-layer statistics
SELECT hnsw_io_stats_reset();

SELECT count(*) AS scan_result_count FROM (
  SELECT id FROM test_layer_stats
  ORDER BY embedding <-> (SELECT embedding FROM test_layer_stats WHERE id = 1)::vector
  LIMIT 10
) sub;

SELECT count(*) > 0 AS scan_has_layer_data FROM hnsw_io_layer_stats();

-- Test 6: GUC off disables per-layer counting
SELECT hnsw_io_stats_reset();
SET hnsw.io_layer_track = off;

INSERT INTO test_layer_stats (embedding)
SELECT (SELECT array_agg(random()::real)::vector FROM generate_series(1,128))
FROM generate_series(1,10);

-- With GUC off, per-layer stats should still be empty
SELECT count(*) = 0 AS guc_off_no_layer_stats FROM hnsw_io_layer_stats();

-- But global counters should still increment
SELECT read_pages > 0 AS global_still_counts FROM hnsw_io_stats();

SET hnsw.io_layer_track = on;

-- Test 7: Cache hit rate sanity check
SELECT hnsw_io_stats_reset();

INSERT INTO test_layer_stats (embedding)
SELECT (SELECT array_agg(random()::real)::vector FROM generate_series(1,128))
FROM generate_series(1,20);

-- Verify nonzero buffer reads
SELECT sum(buffer_read_pages) > 0 AS has_buffer_reads FROM hnsw_io_layer_stats();

-- Cleanup
DROP TABLE test_layer_stats CASCADE;
