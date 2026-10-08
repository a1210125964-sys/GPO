#include "postgres.h"

#include "fmgr.h"
#include "funcapi.h"
#include "hnsw.h"
#include "utils/guc.h"

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
bool		hnsw_tuple_trace = false;
static uint64 hnsw_tuple_trace_query_id = 0;
static uint64 hnsw_tuple_trace_access_order = 0;

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

	DefineCustomBoolVariable("hnsw.tuple_trace",
		"Enable tuple access tracing for the HNSW diagnostic experiment.",
		"When enabled, tuple accesses are emitted as HNSW_TUPLE_TRACE log lines.",
		&hnsw_tuple_trace,
		false,
		PGC_USERSET,
		0,
		NULL,
		NULL,
		NULL);
}

void
HnswTupleTraceStartQuery(void)
{
	if (!hnsw_tuple_trace)
		return;

	hnsw_tuple_trace_query_id++;
	hnsw_tuple_trace_access_order = 0;
}

void
HnswTupleTraceRecord(BlockNumber blkno, OffsetNumber offno,
					 const char *tuple_type, int layer)
{
	uint64		tuple_id;

	if (!hnsw_tuple_trace)
		return;

	/* tuple_id is a stable diagnostic ID derived from (block, offset). */
	tuple_id = ((uint64) blkno << 16) | (uint64) offno;

	elog(LOG,
			"HNSW_TUPLE_TRACE query_id=" UINT64_FORMAT
			" access_order=" UINT64_FORMAT
			" tuple_id=" UINT64_FORMAT
			" tuple_type=%s layer=%d block_number=%u",
			hnsw_tuple_trace_query_id,
			hnsw_tuple_trace_access_order++, tuple_id,
			tuple_type, layer, blkno);
}
