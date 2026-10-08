#include "postgres.h"

#include "fmgr.h"
#include "funcapi.h"
#include "hnsw.h"
#include "utils/guc.h"
#include "storage/bufmgr.h"    /* for InvalidBlockNumber */

/* Per-session I/O counters for HNSW insert operations */
bool		hnsw_io_track = false;
int64		hnsw_io_read_pages = 0;
int64		hnsw_io_write_pages = 0;
int64		hnsw_io_total_ios = 0;
int64		hnsw_io_updated_pages = 0;
HnswIOPageSet *hnsw_io_page_set = NULL;

/* Per-layer read statistics */
int			hnsw_io_current_layer = -1;
int64		hnsw_io_layer_buffer_reads[HNSW_MAX_TRACKED_LEVELS] = {0};
int64		hnsw_io_layer_io_reads[HNSW_MAX_TRACKED_LEVELS] = {0};
bool		hnsw_io_layer_track = true;

/* Per-session sequential/random read statistics */
BlockNumber	hnsw_io_last_block = InvalidBlockNumber;
bool		hnsw_io_seq_rand_track = true;
int64		hnsw_io_seq_buffer_reads = 0;
int64		hnsw_io_seq_io_reads = 0;
int64		hnsw_io_rand_buffer_reads = 0;
int64		hnsw_io_rand_io_reads = 0;
int64		hnsw_io_first_buffer_reads = 0;
int64		hnsw_io_first_io_reads = 0;

/* Per-reason write counters */
int64		hnsw_io_write_reason_counts[HNSW_WRITE_NREASONS] = {0};

/* Human-readable names for SQL output */
const char *hnsw_io_write_reason_names[] = {
	[HNSW_WRITE_ADD_ELEMENT]			= "add_element",
	[HNSW_WRITE_UPDATE_NEIGHBOR]		= "update_neighbor",
	[HNSW_WRITE_UPDATE_META_INSERT_PAGE] = "update_meta_insert_page",
	[HNSW_WRITE_UPDATE_META_ENTRY_POINT] = "update_meta_entry_point",
	[HNSW_WRITE_ADD_DUPLICATE]			= "add_duplicate",
};

/*
 * Mark a page as updated for deduplication tracking.
 * Returns true if the page is new (not previously counted in the current set),
 * false if it was already counted in this statement/build.
 */
bool
HnswIOMarkPageUpdated(BlockNumber blkno)
{
	if (hnsw_io_page_set == NULL || hnsw_io_page_set->overflow)
		return true;

	for (int i = 0; i < hnsw_io_page_set->npages; i++)
	{
		if (hnsw_io_page_set->pages[i] == blkno)
			return false;
	}

	if (hnsw_io_page_set->npages < HNSW_IO_MAX_TRACKED_PAGES)
		hnsw_io_page_set->pages[hnsw_io_page_set->npages++] = blkno;
	else
		hnsw_io_page_set->overflow = true;

	return true;
}

PG_FUNCTION_INFO_V1(hnsw_io_stats);
Datum
hnsw_io_stats(PG_FUNCTION_ARGS)
{
	Datum		values[4];
	bool		nulls[4];
	TupleDesc	tupdesc;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "function returning record called in context that cannot accept type record");

	BlessTupleDesc(tupdesc);

	values[0] = Int64GetDatum(hnsw_io_read_pages);
	nulls[0] = false;
	values[1] = Int64GetDatum(hnsw_io_write_pages);
	nulls[1] = false;
	values[2] = Int64GetDatum(hnsw_io_total_ios);
	nulls[2] = false;
	values[3] = Int64GetDatum(hnsw_io_updated_pages);
	nulls[3] = false;

	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

PG_FUNCTION_INFO_V1(hnsw_io_stats_reset);
Datum
hnsw_io_stats_reset(PG_FUNCTION_ARGS)
{
	hnsw_io_read_pages = 0;
	hnsw_io_write_pages = 0;
	hnsw_io_total_ios = 0;
	hnsw_io_updated_pages = 0;
	memset(hnsw_io_layer_buffer_reads, 0, sizeof(hnsw_io_layer_buffer_reads));
	memset(hnsw_io_layer_io_reads, 0, sizeof(hnsw_io_layer_io_reads));
	memset(hnsw_io_write_reason_counts, 0, sizeof(hnsw_io_write_reason_counts));

	/* Reset seq/rand/first counters and last-block tracker (Rule 5) */
	hnsw_io_last_block = InvalidBlockNumber;
	hnsw_io_seq_buffer_reads = 0;
	hnsw_io_seq_io_reads = 0;
	hnsw_io_rand_buffer_reads = 0;
	hnsw_io_rand_io_reads = 0;
	hnsw_io_first_buffer_reads = 0;
	hnsw_io_first_io_reads = 0;

	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(hnsw_io_write_reasons);
Datum
hnsw_io_write_reasons(PG_FUNCTION_ARGS)
{
	Datum		values[HNSW_WRITE_NREASONS];
	bool		nulls[HNSW_WRITE_NREASONS];
	TupleDesc	tupdesc;
	int			i;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "function returning record called in context that cannot accept type record");

	BlessTupleDesc(tupdesc);

	for (i = 0; i < HNSW_WRITE_NREASONS; i++)
	{
		values[i] = Int64GetDatum(hnsw_io_write_reason_counts[i]);
		nulls[i] = false;
	}

	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

PG_FUNCTION_INFO_V1(hnsw_io_seq_rand_stats);
Datum
hnsw_io_seq_rand_stats(PG_FUNCTION_ARGS)
{
	Datum		values[6];
	bool		nulls[6];
	TupleDesc	tupdesc;

	/*
	 * values[] array MUST match the SQL type column order:
	 *   values[0] = seq_buffer_reads   (hnsw_io_seq_rand_stats_type.seq_buffer_reads)
	 *   values[1] = seq_io_reads       (hnsw_io_seq_rand_stats_type.seq_io_reads)
	 *   values[2] = rand_buffer_reads  (hnsw_io_seq_rand_stats_type.rand_buffer_reads)
	 *   values[3] = rand_io_reads      (hnsw_io_seq_rand_stats_type.rand_io_reads)
	 *   values[4] = first_buffer_reads (hnsw_io_seq_rand_stats_type.first_buffer_reads)
	 *   values[5] = first_io_reads     (hnsw_io_seq_rand_stats_type.first_io_reads)
	 * If the SQL type is reordered, this mapping MUST be updated.
	 */

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "function returning record called in context "
					"that cannot accept type record");

	BlessTupleDesc(tupdesc);

	values[0] = Int64GetDatum(hnsw_io_seq_buffer_reads);   /* seq_buffer_reads */
	nulls[0] = false;
	values[1] = Int64GetDatum(hnsw_io_seq_io_reads);       /* seq_io_reads */
	nulls[1] = false;
	values[2] = Int64GetDatum(hnsw_io_rand_buffer_reads);  /* rand_buffer_reads */
	nulls[2] = false;
	values[3] = Int64GetDatum(hnsw_io_rand_io_reads);      /* rand_io_reads */
	nulls[3] = false;
	values[4] = Int64GetDatum(hnsw_io_first_buffer_reads); /* first_buffer_reads */
	nulls[4] = false;
	values[5] = Int64GetDatum(hnsw_io_first_io_reads);     /* first_io_reads */
	nulls[5] = false;

	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

PG_FUNCTION_INFO_V1(hnsw_io_layer_stats);
Datum
hnsw_io_layer_stats(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext old_ctx;
	int			lc;

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("set-valued function called in context that "
						"cannot accept a set")));
	if (!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("materialize mode required, but it is not "
						"allowed in this context")));

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in context "
						"that cannot accept type record")));

	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	old_ctx = MemoryContextSwitchTo(per_query_ctx);

	tupstore = tuplestore_begin_heap(true, false, work_mem);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	MemoryContextSwitchTo(old_ctx);

	for (lc = 0; lc < HNSW_MAX_TRACKED_LEVELS; lc++)
	{
		Datum		values[3];
		bool		nulls[3];

		/*
		 * Only emit rows for layers that have non-zero counts.
		 * This keeps output concise; layers 0..maxLayer are
		 * typically far fewer than 256.
		 */
		if (hnsw_io_layer_buffer_reads[lc] == 0 &&
			hnsw_io_layer_io_reads[lc] == 0)
			continue;

		values[0] = Int16GetDatum((int16) lc);
		nulls[0] = false;
		values[1] = Int64GetDatum(hnsw_io_layer_io_reads[lc]);
		nulls[1] = false;
		values[2] = Int64GetDatum(hnsw_io_layer_buffer_reads[lc]);
		nulls[2] = false;

		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
	}

	/* tuplestore is kept alive via rsinfo->setResult; caller reads it */
	return (Datum) 0;
}

/* Register HNSW per-layer I/O statistics GUCs */
void
HnswIOStatInit(void)
{
	DefineCustomBoolVariable("hnsw.io_layer_track",
		"Enable per-layer HNSW read page statistics.",
		"When disabled, per-layer counters are not incremented "
		"(global read/write counters are unaffected).",
		&hnsw_io_layer_track,
		true,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);

	DefineCustomBoolVariable("hnsw.io_seq_rand_track",
		"Enable sequential/random HNSW read page classification.",
		"When disabled, seq/rand/first counters stop incrementing "
		"and hnsw_io_last_block is not updated "
		"(previously accumulated values are retained — "
		"use hnsw_io_stats_reset() to clear them). "
		"Global read/write and per-layer counters are unaffected.",
		&hnsw_io_seq_rand_track,
		true,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);
}
