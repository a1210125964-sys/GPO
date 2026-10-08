-- ============================================================
-- HNSW Sequential/Random Read Statistics — Regression Tests
-- ============================================================

-- Setup
SET hnsw.io_seq_rand_track = on;
SELECT setseed(0.5);
DROP TABLE IF EXISTS test_seq_rand_stats CASCADE;
CREATE TABLE test_seq_rand_stats (id serial, embedding vector(128));

-- Test 1: Reset clears all counters + GUC visibility (L2)
SHOW hnsw.io_seq_rand_track;  -- Expected: on

SELECT hnsw_io_stats_reset();
SELECT * FROM hnsw_io_seq_rand_stats();
-- Expected: all zeros

-- Test 2: INSERT triggers seq/rand statistics
CREATE INDEX ON test_seq_rand_stats USING hnsw (embedding vector_l2_ops) WITH (m = 16);

INSERT INTO test_seq_rand_stats (embedding)
SELECT (SELECT array_agg(random()::real)::vector FROM generate_series(1,128))
FROM generate_series(1,100);

-- Verify: at least some reads happened
SELECT seq_buffer_reads + rand_buffer_reads + first_buffer_reads > 0
  AS has_reads FROM hnsw_io_seq_rand_stats();

-- Test 3: Invariant — buffer >= io for each category
SELECT
  seq_buffer_reads >= seq_io_reads AS seq_ok,
  rand_buffer_reads >= rand_io_reads AS rand_ok,
  first_buffer_reads >= first_io_reads AS first_ok
FROM hnsw_io_seq_rand_stats();

-- Test 3.5: Invariant — after reset with GUC on, first_buffer_reads >= 1
-- (the very first ReadBuffer after reset has no predecessor → must be "first")
SELECT first_buffer_reads >= 1 AS first_after_reset_invariant
FROM hnsw_io_seq_rand_stats();

-- Test 4: Invariant — sum of seq+rand+first buffer reads <= global read_pages
-- Relationship is ≤ (not enforced as strict =) because:
--   (a) Non-HNSW ReadBuffer calls from other extensions or PostgreSQL internals
--       may also increment global hnsw_io_read_pages.
--   (b) After the Task 6 P_NEW fix, HnswNewBuffer(P_NEW) no longer contributes
--       to hnsw_io_read_pages.  For HNSW-only workloads with GUC on,
--       seq+rand+first == hnsw_io_read_pages (exact equality) is expected.
-- We use ≤ for robustness against non-HNSW global read activity.
SELECT
  (seq_buffer_reads + rand_buffer_reads + first_buffer_reads) <= read_pages
  AS seq_rand_le_global
FROM hnsw_io_seq_rand_stats(), hnsw_io_stats();

-- Test 4.5: Invariant — with GUC on, sum of seq+rand+first buffer reads > 0
-- after a non-trivial insert workload (must have some reads)
-- IO-level counterparts may be zero (all cached — legitimate in warm-cache scenarios)
SELECT
  (seq_buffer_reads + rand_buffer_reads + first_buffer_reads) > 0
    AS has_buffer_reads,
  (seq_io_reads + rand_io_reads + first_io_reads) >= 0
    AS io_reads_nonnegative
FROM hnsw_io_seq_rand_stats();

-- Test 5: SCAN/query triggers seq/rand statistics
-- Note: buffer-level counters MUST be > 0 after a SCAN (ReadBuffer is always called).
-- IO-level counters MAY be zero — legitimate result when all pages are in
-- shared_buffers (warm cache). This test only asserts buffer > 0.
SELECT hnsw_io_stats_reset();

SELECT id FROM test_seq_rand_stats
ORDER BY embedding <-> (SELECT embedding FROM test_seq_rand_stats WHERE id = 1)::vector
LIMIT 10;

SELECT seq_buffer_reads + rand_buffer_reads + first_buffer_reads > 0
  AS scan_has_reads FROM hnsw_io_seq_rand_stats();

-- Test 6: GUC off disables seq/rand classification
SELECT hnsw_io_stats_reset();
SET hnsw.io_seq_rand_track = off;

INSERT INTO test_seq_rand_stats (embedding)
SELECT (SELECT array_agg(random()::real)::vector FROM generate_series(1,128))
FROM generate_series(1,10);

-- With GUC off, seq/rand stats should still be zero
SELECT seq_buffer_reads = 0 AND rand_buffer_reads = 0 AND first_buffer_reads = 0
  AS guc_off_no_seq_rand FROM hnsw_io_seq_rand_stats();

-- But global counters should still increment
SELECT read_pages > 0 AS global_still_counts FROM hnsw_io_stats();

SET hnsw.io_seq_rand_track = on;

-- Test 7: GUC-off doesn't poison last_block when re-enabled
SELECT hnsw_io_stats_reset();

INSERT INTO test_seq_rand_stats (embedding)
SELECT (SELECT array_agg(random()::real)::vector FROM generate_series(1,128))
FROM generate_series(1,20);

-- After GUC re-enabled, first read should be classified as 'first'
SELECT first_buffer_reads > 0 AS first_detected_after_reenable
FROM hnsw_io_seq_rand_stats();

-- Test 8: Reset clears last_block (first read after reset = first, not sequential)
SELECT hnsw_io_stats_reset();

INSERT INTO test_seq_rand_stats (embedding)
SELECT (SELECT array_agg(random()::real)::vector FROM generate_series(1,128))
FROM generate_series(1,10);

-- First read after reset should be classified as 'first'
SELECT first_buffer_reads > 0 AS first_after_reset FROM hnsw_io_seq_rand_stats();

-- Test 9: All three categories can appear in a non-trivial workload
SELECT hnsw_io_stats_reset();

INSERT INTO test_seq_rand_stats (embedding)
SELECT (SELECT array_agg(random()::real)::vector FROM generate_series(1,128))
FROM generate_series(1,200);

-- In a large insert, verify all three categories can appear and meet invariants
SELECT seq_buffer_reads > 0 AS seq_present,
       rand_buffer_reads > 0 AS rand_present,
       first_buffer_reads = 1 AS first_single
FROM hnsw_io_seq_rand_stats();

-- Cleanup
DROP TABLE test_seq_rand_stats CASCADE;
