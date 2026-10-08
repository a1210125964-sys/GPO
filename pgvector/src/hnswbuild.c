/*
 * The HNSW build happens in two phases:
 *
 * 1. In-memory phase
 *
 * In this first phase, the graph is held completely in memory. When the graph
 * is fully built, or we run out of memory reserved for the build (determined
 * by maintenance_work_mem), we materialize the graph to disk (see
 * FlushPages()), and switch to the on-disk phase.
 *
 * In a parallel build, a large contiguous chunk of shared memory is allocated
 * to hold the graph. Each worker process has its own HnswBuildState struct in
 * private memory, which contains information that doesn't change throughout
 * the build, and pointers to the shared structs in shared memory. The shared
 * memory area is mapped to a different address in each worker process, and
 * 'HnswBuildState.hnswarea' points to the beginning of the shared area in the
 * worker process's address space. All pointers used in the graph are
 * "relative pointers", stored as an offset from 'hnswarea'.
 *
 * Each element is protected by an LWLock. It must be held when reading or
 * modifying the element's neighbors or 'heaptids'.
 *
 * In a non-parallel build, the graph is held in backend-private memory. All
 * the elements are allocated in a dedicated memory context, 'graphCtx', and
 * the pointers used in the graph are regular pointers.
 *
 * 2. On-disk phase
 *
 * In the on-disk phase, the index is built by inserting each vector to the
 * index one by one, just like on INSERT. The only difference is that we don't
 * WAL-log the individual inserts. If the graph fit completely in memory and
 * was fully built in the in-memory phase, the on-disk phase is skipped.
 *
 * After we have finished building the graph, we perform one more scan through
 * the index and write all the pages to the WAL.
 */
#include "postgres.h"

#include <math.h>

#include "access/parallel.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "hnsw.h"
#include "miscadmin.h"
#include "optimizer/optimizer.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "tcop/tcopprot.h"
#include "utils/datum.h"
#include "utils/memutils.h"

#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#else
#include "pgstat.h"
#endif

#if PG_VERSION_NUM >= 140000
#include "utils/backend_status.h"
#include "utils/wait_event.h"
#endif

/* Return elapsed wall-clock time since start in milliseconds. */
static double
AftpBuildElapsedMilliseconds(instr_time start)
{
	instr_time	end;

	INSTR_TIME_SET_CURRENT(end);
	INSTR_TIME_SUBTRACT(end, start);

	return INSTR_TIME_GET_MILLISEC(end);
}

#define PARALLEL_KEY_HNSW_SHARED		UINT64CONST(0xA000000000000001)
#define PARALLEL_KEY_HNSW_AREA			UINT64CONST(0xA000000000000002)
#define PARALLEL_KEY_QUERY_TEXT			UINT64CONST(0xA000000000000003)

/*
 * Create the metapage
 */
static void
CreateMetaPage(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = HnswNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	HnswInitPage(buf, page);

	/* Set metapage data */
	metap = HnswPageGetMeta(page);
	metap->magicNumber = HNSW_MAGIC_NUMBER;
	metap->version = HNSW_VERSION;
	metap->dimensions = buildstate->dimensions;
	metap->m = buildstate->m;
	metap->efConstruction = buildstate->efConstruction;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;
	metap->aftp_build_status = 0;
	metap->aftp_storage_version = 0;
	metap->aftp_n_epages = 0;
	metap->aftp_n_npages = 0;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(HnswMetaPageData)) - (char *) page;

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/*
 * Add a new page
 */
static void
HnswBuildAppendPage(Relation index, Buffer *buf, Page *page, ForkNumber forkNum)
{
	/* Add a new page */
	Buffer		newbuf = HnswNewBuffer(index, forkNum);

	/* Update previous page */
	HnswPageGetOpaque(*page)->nextblkno = BufferGetBlockNumber(newbuf);

	/* Commit */
	MarkBufferDirty(*buf);
	UnlockReleaseBuffer(*buf);

	/* Can take a while, so ensure we can interrupt */
	/* Needs to be called when no buffer locks are held */
	LockBuffer(newbuf, BUFFER_LOCK_UNLOCK);
	CHECK_FOR_INTERRUPTS();
	LockBuffer(newbuf, BUFFER_LOCK_EXCLUSIVE);

	/* Prepare new page */
	*buf = newbuf;
	*page = BufferGetPage(*buf);
	HnswInitPage(*buf, *page);
}

/*
 * Create graph pages — centralized NeighborTuple version
 *
 * Maintains two independent page chains:
 *   1. ElementTuple chain (ebuf/epage) — only ElementTuples
 *   2. NeighborTuple chain (nbuf/npage) — only NeighborTuples
 *
 * ElementTuple.neighbortid → (neighborPage, neighborOffno)
 * points to the centralized NeighborTuple page chain.
 */
static void
CreateGraphPages(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	HnswElementTuple etup;
	HnswNeighborTuple ntup;
	BlockNumber insertPage;
	BlockNumber neighborInsertPage;
	HnswElement entryPoint;
	Buffer		ebuf;		/* ElementTuple page buffer */
	Page		epage;		/* ElementTuple page */
	Buffer		nbuf;		/* NeighborTuple page buffer (centralized pool) */
	Page		npage;		/* NeighborTuple page (centralized pool) */
	HnswElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;

	/* Allocate once */
	etup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
	ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

	/* Prepare first ElementTuple page */
	ebuf = HnswNewBuffer(index, forkNum);
	epage = BufferGetPage(ebuf);
	HnswInitPage(ebuf, epage);

	/* Prepare first NeighborTuple page (centralized pool) */
	nbuf = HnswNewBuffer(index, forkNum);
	npage = BufferGetPage(nbuf);
	HnswInitPage(nbuf, npage);

	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);
		Size		etupSize;
		Size		ntupSize;
		Pointer		valuePtr = HnswPtrAccess(base, element->value);

		/* Update iterator */
		iter = element->next;

		/* Zero memory for each element */
		MemSet(etup, 0, HNSW_TUPLE_ALLOC_SIZE);

		/* Calculate sizes */
		etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(valuePtr));
		ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, buildstate->m);

		/* Initial size check */
		if (etupSize > HNSW_TUPLE_ALLOC_SIZE)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("index tuple too large")));

		HnswSetElementTuple(base, etup, element);

		/*
		 * ElementTuple: write to ElementTuple page chain.
		 * Append new ElementTuple page when out of space.
		 */
		if (PageGetFreeSpace(epage) < etupSize)
			HnswBuildAppendPage(index, &ebuf, &epage, forkNum);

		element->blkno = BufferGetBlockNumber(ebuf);
		element->offno = OffsetNumberNext(PageGetMaxOffsetNumber(epage));

		/*
		 * NeighborTuple: write to centralized NeighborTuple page chain.
		 * Append new NeighborTuple page when out of space.
		 */
		if (PageGetFreeSpace(npage) < ntupSize)
			HnswBuildAppendPage(index, &nbuf, &npage, forkNum);

		element->neighborPage = BufferGetBlockNumber(nbuf);
		element->neighborOffno = OffsetNumberNext(PageGetMaxOffsetNumber(npage));

		ItemPointerSet(&etup->neighbortid, element->neighborPage, element->neighborOffno);

		/* Add element to ElementTuple page */
		if (PageAddItem(epage, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != element->offno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		/* Add neighbor placeholder to centralized NeighborTuple page */
		if (PageAddItem(npage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != element->neighborOffno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}

	insertPage = BufferGetBlockNumber(ebuf);
	neighborInsertPage = BufferGetBlockNumber(nbuf);

	/* Commit ElementTuple page chain */
	MarkBufferDirty(ebuf);
	UnlockReleaseBuffer(ebuf);

	/* Commit NeighborTuple page chain */
	MarkBufferDirty(nbuf);
	UnlockReleaseBuffer(nbuf);

	entryPoint = HnswPtrAccess(base, buildstate->graph->entryPoint);
	/* Pass actual neighborInsertPage to metapage */
	HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_ALWAYS, entryPoint,
	                   insertPage, neighborInsertPage,
	                   forkNum, true, HNSW_WRITE_UPDATE_META_ENTRY_POINT);

	pfree(etup);
	pfree(ntup);
}

/* ----------------------------------------------------------------
 * Backfill sort entry for Pass 3 group-by-blkno processing.
 * ---------------------------------------------------------------- */
typedef struct
{
	BlockNumber blkno;
	OffsetNumber offno;
	BlockNumber neighborPage;
	OffsetNumber neighborOffno;
}			BackfillSortEntry;

/* Comparison function for sorting BackfillSortEntry by blkno */
static int
compare_backfill_by_blkno(const void *a, const void *b)
{
	BlockNumber ba = ((const BackfillSortEntry *) a)->blkno;
	BlockNumber bb = ((const BackfillSortEntry *) b)->blkno;

	if (ba < bb)
		return -1;
	if (ba > bb)
		return 1;
	return 0;
}

/*
 * Create graph pages in AFTP-optimized packed order
 *
 * Three passes:
 *   Pass 1: Write ElementTuples in aftp_etuple_order[]
 *   Pass 2: Write NeighborTuple placeholders in aftp_ntuple_order[]
 *   Pass 3: Backfill ElementTuple.neighbortid, grouped by blkno
 *
 * Requires that AftpMain() has already populated the aftp_* fields
 * in buildstate.  After this function, WriteNeighborTuples() fills
 * the actual neighbor data into the placeholder slots.
 */
static void
CreateGraphPagesPacked(HnswBuildState *buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	HnswElementTuple etup;
	HnswNeighborTuple ntup;
	Buffer		ebuf;
	Page		epage;
	Buffer		nbuf;
	Page		npage;
	HnswElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;
	int			n_elements;
	int			n_neighbors;
	HnswElement *id_to_element;
	BlockNumber insertPage;
	BlockNumber neighborInsertPage;
	instr_time	packed_start;
	instr_time	pass_start;
	double		pass1_ms;
	double		pass2_ms;
	double		pass3_ms;

	/* ---- Count elements and build id_to_element mapping ---- */
	n_elements = 0;
	while (!HnswPtrIsNull(base, iter))
	{
		n_elements++;
		iter = HnswPtrAccess(base, iter)->next;
	}
	n_neighbors = n_elements;

	if (n_elements == 0)
		return;

	INSTR_TIME_SET_CURRENT(packed_start);

	id_to_element = palloc(n_elements * sizeof(HnswElement));
	iter = buildstate->graph->head;
	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);
		id_to_element[element->aftp_tuple_id] = element;
		iter = element->next;
	}

	/* Allocate once */
	etup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
	ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

	/****************************************************
	 * Pass 1: Write ElementTuples in packed order
	 ****************************************************/
	INSTR_TIME_SET_CURRENT(pass_start);
	{
		BlockNumber *etuple_blkno;
		OffsetNumber *etuple_offno;

		ebuf = HnswNewBuffer(index, forkNum);
		epage = BufferGetPage(ebuf);
		HnswInitPage(ebuf, epage);

		etuple_blkno = palloc0(n_elements * sizeof(BlockNumber));
		etuple_offno = palloc0(n_elements * sizeof(OffsetNumber));

		for (int i = 0; i < n_elements; i++)
		{
			int			tid = buildstate->aftp_etuple_order[i];
			HnswElement element = id_to_element[tid];
			Size		etupSize;
			Pointer		valuePtr = HnswPtrAccess(base, element->value);

			MemSet(etup, 0, HNSW_TUPLE_ALLOC_SIZE);

			etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(valuePtr));

			if (etupSize > HNSW_TUPLE_ALLOC_SIZE)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("index tuple too large")));

			HnswSetElementTuple(base, etup, element);

			if (PageGetFreeSpace(epage) < etupSize)
				HnswBuildAppendPage(index, &ebuf, &epage, forkNum);

			element->blkno = BufferGetBlockNumber(ebuf);
			element->offno = OffsetNumberNext(PageGetMaxOffsetNumber(epage));

			etuple_blkno[tid] = element->blkno;
			etuple_offno[tid] = element->offno;

			if (PageAddItem(epage, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != element->offno)
				elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
		}

		insertPage = BufferGetBlockNumber(ebuf);
		MarkBufferDirty(ebuf);
		UnlockReleaseBuffer(ebuf);

		pfree(etuple_blkno);
		pfree(etuple_offno);
	}
	pass1_ms = AftpBuildElapsedMilliseconds(pass_start);
	elog(INFO, "AFTP timing: write Pass 1 ElementTuples: %.3f ms",
		 pass1_ms);

	/****************************************************
	 * Pass 2: Write NeighborTuple placeholders in packed order
	 ****************************************************/
	INSTR_TIME_SET_CURRENT(pass_start);
	{
		nbuf = HnswNewBuffer(index, forkNum);
		npage = BufferGetPage(nbuf);
		HnswInitPage(nbuf, npage);

		for (int i = 0; i < n_neighbors; i++)
		{
			int			tid = buildstate->aftp_ntuple_order[i];
			HnswElement element = id_to_element[tid];
			Size		ntupSize;

			MemSet(ntup, 0, HNSW_TUPLE_ALLOC_SIZE);
			ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, buildstate->m);

			if (PageGetFreeSpace(npage) < ntupSize)
				HnswBuildAppendPage(index, &nbuf, &npage, forkNum);

			element->neighborPage = BufferGetBlockNumber(nbuf);
			element->neighborOffno = OffsetNumberNext(PageGetMaxOffsetNumber(npage));

			if (PageAddItem(npage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != element->neighborOffno)
				elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
		}

		neighborInsertPage = BufferGetBlockNumber(nbuf);
		MarkBufferDirty(nbuf);
		UnlockReleaseBuffer(nbuf);
	}
	pass2_ms = AftpBuildElapsedMilliseconds(pass_start);
	elog(INFO, "AFTP timing: write Pass 2 NeighborTuple placeholders: %.3f ms",
		 pass2_ms);

	/****************************************************
	 * Pass 3: Backfill ElementTuple.neighbortid
	 *
	 * Group by blkno, one ReadBuffer per page.
	 ****************************************************/
	INSTR_TIME_SET_CURRENT(pass_start);
	{
		BackfillSortEntry *entries;
		int			idx;

		entries = palloc(n_elements * sizeof(BackfillSortEntry));

		for (int i = 0; i < n_elements; i++)
		{
			HnswElement element = id_to_element[i];

			entries[i].blkno = element->blkno;
			entries[i].offno = element->offno;
			entries[i].neighborPage = element->neighborPage;
			entries[i].neighborOffno = element->neighborOffno;
		}

		/* Sort by blkno so we can process in page groups */
		qsort(entries, n_elements, sizeof(BackfillSortEntry),
			  compare_backfill_by_blkno);

		idx = 0;
		while (idx < n_elements)
		{
			BlockNumber blkno = entries[idx].blkno;
			Buffer		buf;
			Page		page;

			buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			page = BufferGetPage(buf);

			while (idx < n_elements && entries[idx].blkno == blkno)
			{
				HnswElementTuple cur_etup;

				cur_etup = (HnswElementTuple) PageGetItem(page,
													  PageGetItemId(page, entries[idx].offno));
				ItemPointerSet(&cur_etup->neighbortid,
							   entries[idx].neighborPage,
							   entries[idx].neighborOffno);
				idx++;
			}

			MarkBufferDirty(buf);
			UnlockReleaseBuffer(buf);
		}

		pfree(entries);
	}
	pass3_ms = AftpBuildElapsedMilliseconds(pass_start);
	elog(INFO, "AFTP timing: write Pass 3 neighbortid backfill: %.3f ms",
		 pass3_ms);

	/* Update metapage */
	{
		HnswElement entryPoint;

		entryPoint = HnswPtrAccess(base, buildstate->graph->entryPoint);
		HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_ALWAYS, entryPoint,
						   insertPage, neighborInsertPage,
						   forkNum, true, HNSW_WRITE_UPDATE_META_ENTRY_POINT);
	}

	pfree(etup);
	pfree(ntup);
	pfree(id_to_element);

	elog(INFO, "AFTP timing: packed write passes total: %.3f ms "
		 "(Pass 1 + Pass 2 + Pass 3: %.3f ms)",
		 AftpBuildElapsedMilliseconds(packed_start),
		 pass1_ms + pass2_ms + pass3_ms);
}

/*
 * Write neighbor tuples
 */
static void
WriteNeighborTuples(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	int			m = buildstate->m;
	HnswElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;
	HnswNeighborTuple ntup;

	/* Allocate once */
	ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);
		Buffer		buf;
		Page		page;
		Size		ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, m);
		int64		io_before = 0;

		/* Update iterator */
		iter = element->next;

		/* Zero memory for each element */
		MemSet(ntup, 0, HNSW_TUPLE_ALLOC_SIZE);

		/* Can take a while, so ensure we can interrupt */
		/* Needs to be called when no buffer locks are held */
		CHECK_FOR_INTERRUPTS();

		if (hnsw_io_track)
			HnswIOCountLayerReadBegin(&io_before);
		buf = ReadBufferExtended(index, forkNum, element->neighborPage, RBM_NORMAL, NULL);
		if (hnsw_io_track)
			HnswIOCountLayerReadEnd(io_before, element->neighborPage);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);

		HnswSetNeighborTuple(base, ntup, element, m);

		if (!PageIndexTupleOverwrite(page, element->neighborOffno, (Item) ntup, ntupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		/* Commit */
		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);
	}

	pfree(ntup);
}

/*
 * Flush pages — AFTP-integrated version
 *
 * When AFTP is enabled and the dataset is large enough (>= 1000 tuples),
 * runs the full AFTP pipeline (Phases 1-4), then writes pages in
 * packed order via CreateGraphPagesPacked().
 *
 * On AFTP failure, falls back to the original sequential layout.
 */
static void
FlushPages(HnswBuildState * buildstate)
{
	bool		use_aftp = aftp_enable && buildstate->graph->indtuples >= 1000;
	bool		packed_layout_written = false;
	instr_time	aftp_total_start;
	instr_time	write_start;

#ifdef HNSW_MEMORY
	elog(INFO, "memory: %zu MB", buildstate->graph->memoryUsed / (1024 * 1024));
#endif

	CreateMetaPage(buildstate);

	if (use_aftp)
	{
		MemoryContext oldCtx;

		INSTR_TIME_SET_CURRENT(aftp_total_start);

		/* Create AFTP dedicated MemoryContext */
		buildstate->aftp_context = AllocSetContextCreate(
			buildstate->graphCtx,
			"AFTP build context",
			ALLOCSET_DEFAULT_SIZES);

		oldCtx = MemoryContextSwitchTo(buildstate->aftp_context);

		PG_TRY();
		{
			AftpConfig cfg;

			/* Mark AFTP in progress in metapage */
			buildstate->aftp_status = AFTP_BUILD_IN_PROGRESS;

			/* Populate config and run AFTP pipeline */
			AftpPopulateConfig(&cfg);
			AftpMain(buildstate, &cfg);

			/* Mark flushing in progress */
			buildstate->aftp_status = AFTP_BUILD_FLUSHING;

			/* Write pages in packed order */
			CreateGraphPagesPacked(buildstate);
			packed_layout_written = true;

			/* Mark complete */
			buildstate->aftp_status = AFTP_BUILD_COMPLETED;
		}
		PG_CATCH();
		{
			/* Clean up AFTP resources */
			MemoryContextSwitchTo(oldCtx);
			MemoryContextDelete(buildstate->aftp_context);
			buildstate->aftp_etuple_page = NULL;
			buildstate->aftp_ntuple_page = NULL;
			buildstate->aftp_etuple_order = NULL;
			buildstate->aftp_ntuple_order = NULL;
			buildstate->aftp_context = NULL;

			/* Fall back to original layout */
			elog(WARNING, "AFTP build failed, falling back to default layout");
			buildstate->aftp_status = AFTP_BUILD_NOT_STARTED;

			/* Flush error state */
			FlushErrorState();

			/* Use original CreateGraphPages */
			CreateGraphPages(buildstate);
		}
		PG_END_TRY();

		/* Switch back and clean up */
		MemoryContextSwitchTo(oldCtx);
		if (buildstate->aftp_context != NULL)
			MemoryContextDelete(buildstate->aftp_context);
		buildstate->aftp_context = NULL;
	}
	else
	{
		/* AFTP disabled or dataset too small: use original layout */
		CreateGraphPages(buildstate);
	}

	if (packed_layout_written)
		INSTR_TIME_SET_CURRENT(write_start);

	WriteNeighborTuples(buildstate);

	if (packed_layout_written)
	{
		elog(INFO, "AFTP timing: NeighborTuple payload fill: %.3f ms",
			 AftpBuildElapsedMilliseconds(write_start));
		elog(INFO, "AFTP timing: end-to-end optimization and write: %.3f ms",
			 AftpBuildElapsedMilliseconds(aftp_total_start));
	}

	buildstate->graph->flushed = true;
	MemoryContextReset(buildstate->graphCtx);
}

/*
 * Add a heap TID to an existing element
 */
static bool
AddDuplicateInMemory(HnswElement element, HnswElement dup)
{
	LWLockAcquire(&dup->lock, LW_EXCLUSIVE);

	if (dup->heaptidsLength == HNSW_HEAPTIDS)
	{
		LWLockRelease(&dup->lock);
		return false;
	}

	HnswAddHeapTid(dup, &element->heaptids[0]);

	LWLockRelease(&dup->lock);

	return true;
}

/*
 * Find duplicate element
 */
static bool
FindDuplicateInMemory(char *base, HnswElement element)
{
	HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, 0);
	Datum		value = HnswGetValue(base, element);

	for (int i = 0; i < neighbors->length; i++)
	{
		HnswCandidate *neighbor = &neighbors->items[i];
		HnswElement neighborElement = HnswPtrAccess(base, neighbor->element);
		Datum		neighborValue = HnswGetValue(base, neighborElement);

		/* Exit early since ordered by distance */
		if (!datumIsEqual(value, neighborValue, false, -1))
			return false;

		/* Check for space */
		if (AddDuplicateInMemory(element, neighborElement))
			return true;
	}

	return false;
}

/*
 * Add to element list
 */
static void
AddElementInMemory(char *base, HnswGraph * graph, HnswElement element)
{
	SpinLockAcquire(&graph->lock);
	element->next = graph->head;
	HnswPtrStore(base, graph->head, element);
	SpinLockRelease(&graph->lock);
}

/*
 * Update neighbors
 */
static void
UpdateNeighborsInMemory(char *base, HnswSupport * support, HnswElement e, int m)
{
	for (int lc = e->level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		Size		neighborsSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		HnswNeighborArray *neighbors = palloc(neighborsSize);

		/* Copy neighbors to local memory */
		LWLockAcquire(&e->lock, LW_SHARED);
		memcpy(neighbors, HnswGetNeighbors(base, e, lc), neighborsSize);
		LWLockRelease(&e->lock);

		for (int i = 0; i < neighbors->length; i++)
		{
			HnswCandidate *hc = &neighbors->items[i];
			HnswElement neighborElement = HnswPtrAccess(base, hc->element);

			/* Keep scan-build happy on Mac x86-64 */
			Assert(neighborElement);

			LWLockAcquire(&neighborElement->lock, LW_EXCLUSIVE);
			HnswUpdateConnection(base, HnswGetNeighbors(base, neighborElement, lc), e, hc->distance, lm, NULL, NULL, support);
			LWLockRelease(&neighborElement->lock);
		}
	}
}

/*
 * Update graph in memory
 */
static void
UpdateGraphInMemory(HnswSupport * support, HnswElement element, int m, int efConstruction, HnswElement entryPoint, HnswBuildState * buildstate)
{
	HnswGraph  *graph = buildstate->graph;
	char	   *base = buildstate->hnswarea;

	/* Look for duplicate */
	if (FindDuplicateInMemory(base, element))
		return;

	/* Add element */
	AddElementInMemory(base, graph, element);

	/* Update neighbors */
	UpdateNeighborsInMemory(base, support, element, m);

	/* Update entry point if needed (already have lock) */
	if (entryPoint == NULL || element->level > entryPoint->level)
		HnswPtrStore(base, graph->entryPoint, element);
}

/*
 * Insert tuple in memory
 */
static void
InsertTupleInMemory(HnswBuildState * buildstate, HnswElement element)
{
	HnswGraph  *graph = buildstate->graph;
	HnswSupport *support = &buildstate->support;
	HnswElement entryPoint;
	LWLock	   *entryLock = &graph->entryLock;
	LWLock	   *entryWaitLock = &graph->entryWaitLock;
	int			efConstruction = buildstate->efConstruction;
	int			m = buildstate->m;
	char	   *base = buildstate->hnswarea;

	/* Wait if another process needs exclusive lock on entry lock */
	LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
	LWLockRelease(entryWaitLock);

	/* Get entry point */
	LWLockAcquire(entryLock, LW_SHARED);
	entryPoint = HnswPtrAccess(base, graph->entryPoint);

	/* Prevent concurrent inserts when likely updating entry point */
	if (entryPoint == NULL || element->level > entryPoint->level)
	{
		/* Release shared lock */
		LWLockRelease(entryLock);

		/* Tell other processes to wait and get exclusive lock */
		LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
		LWLockAcquire(entryLock, LW_EXCLUSIVE);
		LWLockRelease(entryWaitLock);

		/* Get latest entry point after lock is acquired */
		entryPoint = HnswPtrAccess(base, graph->entryPoint);
	}

	/* Find neighbors for element */
	HnswFindElementNeighbors(base, element, entryPoint, NULL, support, m, efConstruction, false);

	/* Update graph in memory */
	UpdateGraphInMemory(support, element, m, efConstruction, entryPoint, buildstate);

	/* Release entry lock */
	LWLockRelease(entryLock);
}

/*
 * Insert tuple
 */
static bool
InsertTuple(Relation index, Datum *values, bool *isnull, ItemPointer heaptid, HnswBuildState * buildstate)
{
	HnswGraph  *graph = buildstate->graph;
	HnswElement element;
	HnswAllocator *allocator = &buildstate->allocator;
	HnswSupport *support = &buildstate->support;
	Size		valueSize;
	Pointer		valuePtr;
	LWLock	   *flushLock = &graph->flushLock;
	char	   *base = buildstate->hnswarea;
	Datum		value;

	/* Form index value */
	if (!HnswFormIndexValue(&value, values, isnull, buildstate->typeInfo, support))
		return false;

	/* Get datum size */
	valueSize = VARSIZE_ANY(DatumGetPointer(value));

	/* Ensure graph not flushed when inserting */
	LWLockAcquire(flushLock, LW_SHARED);

	/* Are we in the on-disk phase? */
	if (graph->flushed)
	{
		LWLockRelease(flushLock);

		return HnswInsertTupleOnDisk(index, support, value, heaptid, true);
	}

	/*
	 * In a parallel build, the HnswElement is allocated from the shared
	 * memory area, so we need to coordinate with other processes.
	 */
	LWLockAcquire(&graph->allocatorLock, LW_EXCLUSIVE);

	/*
	 * Check that we have enough memory available for the new element now that
	 * we have the allocator lock, and flush pages if needed.
	 */
	if (graph->memoryUsed >= graph->memoryTotal)
	{
		LWLockRelease(&graph->allocatorLock);

		LWLockRelease(flushLock);
		LWLockAcquire(flushLock, LW_EXCLUSIVE);

		if (!graph->flushed)
		{
			ereport(NOTICE,
					(errmsg("hnsw graph no longer fits into maintenance_work_mem after " INT64_FORMAT " tuples", (int64) graph->indtuples),
					 errdetail("Building will take significantly more time."),
					 errhint("Increase maintenance_work_mem to speed up builds.")));

			FlushPages(buildstate);
		}

		LWLockRelease(flushLock);

		return HnswInsertTupleOnDisk(index, support, value, heaptid, true);
	}

	/* Ok, we can proceed to allocate the element */
	element = HnswInitElement(base, heaptid, buildstate->m, buildstate->ml, buildstate->maxLevel, allocator);
	valuePtr = HnswAlloc(allocator, valueSize);

	/*
	 * We have now allocated the space needed for the element, so we don't
	 * need the allocator lock anymore. Release it and initialize the rest of
	 * the element.
	 */
	LWLockRelease(&graph->allocatorLock);

	/* Copy the datum */
	memcpy(valuePtr, DatumGetPointer(value), valueSize);
	HnswPtrStore(base, element->value, valuePtr);

	/* Create a lock for the element */
	LWLockInitialize(&element->lock, hnsw_lock_tranche_id);

	/* Insert tuple */
	InsertTupleInMemory(buildstate, element);

	/* Release flush lock */
	LWLockRelease(flushLock);

	return true;
}

/*
 * Callback for table_index_build_scan
 */
static void
BuildCallback(Relation index, ItemPointer tid, Datum *values,
			  bool *isnull, bool tupleIsAlive, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	HnswGraph  *graph = buildstate->graph;
	MemoryContext oldCtx;

	/* Skip nulls */
	if (isnull[0])
		return;

	/* Use memory context */
	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	/* Insert tuple */
	if (InsertTuple(index, values, isnull, tid, buildstate))
	{
		/* Update progress */
		SpinLockAcquire(&graph->lock);
		pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, ++graph->indtuples);
		SpinLockRelease(&graph->lock);
	}

	/* Reset memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/*
 * Initialize the graph
 */
static void
InitGraph(HnswGraph * graph, char *base, Size memoryTotal)
{
	/* Initialize the lock tranche if needed */
	HnswInitLockTranche();

	HnswPtrStore(base, graph->head, (HnswElement) NULL);
	HnswPtrStore(base, graph->entryPoint, (HnswElement) NULL);
	graph->memoryUsed = 0;
	graph->memoryTotal = memoryTotal;
	graph->flushed = false;
	graph->indtuples = 0;
	SpinLockInit(&graph->lock);
	LWLockInitialize(&graph->entryLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->entryWaitLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->allocatorLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->flushLock, hnsw_lock_tranche_id);
}

/*
 * Initialize an allocator
 */
static void
InitAllocator(HnswAllocator * allocator, void *(*alloc) (Size size, void *state), void *state)
{
	allocator->alloc = alloc;
	allocator->state = state;
}

/*
 * Memory context allocator
 */
static void *
HnswMemoryContextAlloc(Size size, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	void	   *chunk = MemoryContextAlloc(buildstate->graphCtx, size);

	buildstate->graphData.memoryUsed = MemoryContextMemAllocated(buildstate->graphCtx, false);

	return chunk;
}

/*
 * Shared memory allocator
 */
static void *
HnswSharedMemoryAlloc(Size size, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	void	   *chunk = buildstate->hnswarea + buildstate->graph->memoryUsed;

	buildstate->graph->memoryUsed += MAXALIGN(size);
	return chunk;
}

/*
 * Initialize the build state
 */
static void
InitBuildState(HnswBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo, ForkNumber forkNum)
{
	buildstate->heap = heap;
	buildstate->index = index;
	buildstate->indexInfo = indexInfo;
	buildstate->forkNum = forkNum;
	buildstate->typeInfo = HnswGetTypeInfo(index);

	buildstate->m = HnswGetM(index);
	buildstate->efConstruction = HnswGetEfConstruction(index);
	buildstate->dimensions = TupleDescAttr(index->rd_att, 0)->atttypmod;

	/* Disallow varbit since require fixed dimensions */
	if (TupleDescAttr(index->rd_att, 0)->atttypid == VARBITOID)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("type not supported for hnsw index")));

	/* Require column to have dimensions to be indexed */
	if (buildstate->dimensions < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));

	if (buildstate->dimensions > buildstate->typeInfo->maxDimensions)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("column cannot have more than %d dimensions for hnsw index", buildstate->typeInfo->maxDimensions)));

	if (buildstate->efConstruction < 2 * buildstate->m)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ef_construction must be greater than or equal to 2 * m")));

	buildstate->reltuples = 0;
	buildstate->indtuples = 0;

	/* Get support functions */
	HnswInitSupport(&buildstate->support, index);

	InitGraph(&buildstate->graphData, NULL, (Size) maintenance_work_mem * 1024L);
	buildstate->graph = &buildstate->graphData;
	buildstate->ml = HnswGetMl(buildstate->m);
	buildstate->maxLevel = HnswGetMaxLevel(buildstate->m);

	buildstate->graphCtx = GenerationContextCreate(CurrentMemoryContext,
												   "Hnsw build graph context",
#if PG_VERSION_NUM >= 150000
												   1024 * 1024, 1024 * 1024,
#endif
												   1024 * 1024);
	buildstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											   "Hnsw build temporary context",
											   ALLOCSET_DEFAULT_SIZES);

	InitAllocator(&buildstate->allocator, &HnswMemoryContextAlloc, buildstate);

	buildstate->hnswleader = NULL;
	buildstate->hnswshared = NULL;
	buildstate->hnswarea = NULL;
}

/*
 * Free resources
 */
static void
FreeBuildState(HnswBuildState * buildstate)
{
	MemoryContextDelete(buildstate->graphCtx);
	MemoryContextDelete(buildstate->tmpCtx);
}

/*
 * Within leader, wait for end of heap scan
 */
static double
ParallelHeapScan(HnswBuildState * buildstate)
{
	HnswShared *hnswshared = buildstate->hnswleader->hnswshared;
	int			nparticipanttuplesorts;
	double		reltuples;

	nparticipanttuplesorts = buildstate->hnswleader->nparticipanttuplesorts;
	for (;;)
	{
		SpinLockAcquire(&hnswshared->mutex);
		if (hnswshared->nparticipantsdone == nparticipanttuplesorts)
		{
			buildstate->graph = &hnswshared->graphData;
			buildstate->hnswarea = buildstate->hnswleader->hnswarea;
			reltuples = hnswshared->reltuples;
			SpinLockRelease(&hnswshared->mutex);
			break;
		}
		SpinLockRelease(&hnswshared->mutex);

		ConditionVariableSleep(&hnswshared->workersdonecv,
							   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}

	ConditionVariableCancelSleep();

	return reltuples;
}

/*
 * Perform a worker's portion of a parallel insert
 */
static void
HnswParallelScanAndInsert(Relation heapRel, Relation indexRel, HnswShared * hnswshared, char *hnswarea, bool progress)
{
	HnswBuildState buildstate;
	TableScanDesc scan;
	double		reltuples;
	IndexInfo  *indexInfo;

	/* Join parallel scan */
	indexInfo = BuildIndexInfo(indexRel);
	indexInfo->ii_Concurrent = hnswshared->isconcurrent;

	/* Init page dedup set for this parallel worker */
	indexInfo->ii_AmCache = palloc0(sizeof(HnswIOPageSet));
	hnsw_io_page_set = (HnswIOPageSet *) indexInfo->ii_AmCache;

	InitBuildState(&buildstate, heapRel, indexRel, indexInfo, MAIN_FORKNUM);
	buildstate.graph = &hnswshared->graphData;
	buildstate.hnswarea = hnswarea;
	InitAllocator(&buildstate.allocator, &HnswSharedMemoryAlloc, &buildstate);
	scan = table_beginscan_parallel(heapRel,
									ParallelTableScanFromHnswShared(hnswshared));
	reltuples = table_index_build_scan(heapRel, indexRel, indexInfo,
									   true, progress, BuildCallback,
									   (void *) &buildstate, scan);

	/* Record statistics */
	SpinLockAcquire(&hnswshared->mutex);
	hnswshared->nparticipantsdone++;
	hnswshared->reltuples += reltuples;
	SpinLockRelease(&hnswshared->mutex);

	/* Log statistics */
	if (progress)
		ereport(DEBUG1, (errmsg("leader processed " INT64_FORMAT " tuples", (int64) reltuples)));
	else
		ereport(DEBUG1, (errmsg("worker processed " INT64_FORMAT " tuples", (int64) reltuples)));

	/* Notify leader */
	ConditionVariableSignal(&hnswshared->workersdonecv);

	hnsw_io_page_set = NULL;
	FreeBuildState(&buildstate);
}

/*
 * Perform work within a launched parallel process
 */
void
HnswParallelBuildMain(dsm_segment *seg, shm_toc *toc)
{
	char	   *sharedquery;
	HnswShared *hnswshared;
	char	   *hnswarea;
	Relation	heapRel;
	Relation	indexRel;
	LOCKMODE	heapLockmode;
	LOCKMODE	indexLockmode;

	/* Set debug_query_string for individual workers first */
	sharedquery = shm_toc_lookup(toc, PARALLEL_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;

	/* Report the query string from leader */
	pgstat_report_activity(STATE_RUNNING, debug_query_string);

	/* Look up shared state */
	hnswshared = shm_toc_lookup(toc, PARALLEL_KEY_HNSW_SHARED, false);

	/* Open relations using lock modes known to be obtained by index.c */
	if (!hnswshared->isconcurrent)
	{
		heapLockmode = ShareLock;
		indexLockmode = AccessExclusiveLock;
	}
	else
	{
		heapLockmode = ShareUpdateExclusiveLock;
		indexLockmode = RowExclusiveLock;
	}

	/* Open relations within worker */
	heapRel = table_open(hnswshared->heaprelid, heapLockmode);
	indexRel = index_open(hnswshared->indexrelid, indexLockmode);

	hnswarea = shm_toc_lookup(toc, PARALLEL_KEY_HNSW_AREA, false);

	/* Perform inserts */
	HnswParallelScanAndInsert(heapRel, indexRel, hnswshared, hnswarea, false);

	/* Close relations within worker */
	index_close(indexRel, indexLockmode);
	table_close(heapRel, heapLockmode);
}

/*
 * End parallel build
 */
static void
HnswEndParallel(HnswLeader * hnswleader)
{
	/* Shutdown worker processes */
	WaitForParallelWorkersToFinish(hnswleader->pcxt);

	/* Free last reference to MVCC snapshot, if one was used */
	if (IsMVCCSnapshot(hnswleader->snapshot))
		UnregisterSnapshot(hnswleader->snapshot);
	DestroyParallelContext(hnswleader->pcxt);
	ExitParallelMode();
}

/*
 * Return size of shared memory required for parallel index build
 */
static Size
ParallelEstimateShared(Relation heap, Snapshot snapshot)
{
	return add_size(BUFFERALIGN(sizeof(HnswShared)), table_parallelscan_estimate(heap, snapshot));
}

/*
 * Within leader, participate as a parallel worker
 */
static void
HnswLeaderParticipateAsWorker(HnswBuildState * buildstate)
{
	HnswLeader *hnswleader = buildstate->hnswleader;

	/* Perform work common to all participants */
	HnswParallelScanAndInsert(buildstate->heap, buildstate->index, hnswleader->hnswshared, hnswleader->hnswarea, true);
}

/*
 * Begin parallel build
 */
static void
HnswBeginParallel(HnswBuildState * buildstate, bool isconcurrent, int request)
{
	ParallelContext *pcxt;
	Snapshot	snapshot;
	Size		esthnswshared;
	Size		esthnswarea;
	Size		estother;
	HnswShared *hnswshared;
	char	   *hnswarea;
	HnswLeader *hnswleader = (HnswLeader *) palloc0(sizeof(HnswLeader));
	bool		leaderparticipates = true;
	int			querylen;

#ifdef DISABLE_LEADER_PARTICIPATION
	leaderparticipates = false;
#endif

	/* Enter parallel mode and create context */
	EnterParallelMode();
	Assert(request > 0);
	pcxt = CreateParallelContext("vector", "HnswParallelBuildMain", request);

	/* Get snapshot for table scan */
	if (!isconcurrent)
		snapshot = SnapshotAny;
	else
		snapshot = RegisterSnapshot(GetTransactionSnapshot());

	/* Estimate size of workspaces */
	esthnswshared = ParallelEstimateShared(buildstate->heap, snapshot);
	shm_toc_estimate_chunk(&pcxt->estimator, esthnswshared);

	/* Leave space for other objects in shared memory */
	/* Docker has a default limit of 64 MB for shm_size */
	/* which happens to be the default value of maintenance_work_mem */
	esthnswarea = maintenance_work_mem * 1024L;
	estother = 3 * 1024 * 1024;
	if (esthnswarea > estother)
		esthnswarea -= estother;

	shm_toc_estimate_chunk(&pcxt->estimator, esthnswarea);
	shm_toc_estimate_keys(&pcxt->estimator, 2);

	/* Finally, estimate PARALLEL_KEY_QUERY_TEXT space */
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
		shm_toc_estimate_keys(&pcxt->estimator, 1);
	}
	else
		querylen = 0;			/* keep compiler quiet */

	/* Everyone's had a chance to ask for space, so now create the DSM */
	InitializeParallelDSM(pcxt);

	/* If no DSM segment was available, back out (do serial build) */
	if (pcxt->seg == NULL)
	{
		if (IsMVCCSnapshot(snapshot))
			UnregisterSnapshot(snapshot);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return;
	}

	/* Store shared build state, for which we reserved space */
	hnswshared = (HnswShared *) shm_toc_allocate(pcxt->toc, esthnswshared);
	/* Initialize immutable state */
	hnswshared->heaprelid = RelationGetRelid(buildstate->heap);
	hnswshared->indexrelid = RelationGetRelid(buildstate->index);
	hnswshared->isconcurrent = isconcurrent;
	ConditionVariableInit(&hnswshared->workersdonecv);
	SpinLockInit(&hnswshared->mutex);
	/* Initialize mutable state */
	hnswshared->nparticipantsdone = 0;
	hnswshared->reltuples = 0;
	table_parallelscan_initialize(buildstate->heap,
								  ParallelTableScanFromHnswShared(hnswshared),
								  snapshot);

	hnswarea = (char *) shm_toc_allocate(pcxt->toc, esthnswarea);
	/* Report less than allocated so never fails */
	InitGraph(&hnswshared->graphData, hnswarea, esthnswarea - 1024 * 1024);

	/*
	 * Avoid base address for relptr for Postgres < 14.5
	 * https://github.com/postgres/postgres/commit/7201cd18627afc64850537806da7f22150d1a83b
	 */
#if PG_VERSION_NUM < 140005
	hnswshared->graphData.memoryUsed += MAXALIGN(1);
#endif

	shm_toc_insert(pcxt->toc, PARALLEL_KEY_HNSW_SHARED, hnswshared);
	shm_toc_insert(pcxt->toc, PARALLEL_KEY_HNSW_AREA, hnswarea);

	/* Store query string for workers */
	if (debug_query_string)
	{
		char	   *sharedquery;

		sharedquery = (char *) shm_toc_allocate(pcxt->toc, querylen + 1);
		memcpy(sharedquery, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_QUERY_TEXT, sharedquery);
	}

	/* Launch workers, saving status for leader/caller */
	LaunchParallelWorkers(pcxt);
	hnswleader->pcxt = pcxt;
	hnswleader->nparticipanttuplesorts = pcxt->nworkers_launched;
	if (leaderparticipates)
		hnswleader->nparticipanttuplesorts++;
	hnswleader->hnswshared = hnswshared;
	hnswleader->snapshot = snapshot;
	hnswleader->hnswarea = hnswarea;

	/* If no workers were successfully launched, back out (do serial build) */
	if (pcxt->nworkers_launched == 0)
	{
		HnswEndParallel(hnswleader);
		return;
	}

	/* Log participants */
	ereport(DEBUG1, (errmsg("using %d parallel workers", pcxt->nworkers_launched)));

	/* Save leader state now that it's clear build will be parallel */
	buildstate->hnswleader = hnswleader;

	/* Join heap scan ourselves */
	if (leaderparticipates)
		HnswLeaderParticipateAsWorker(buildstate);

	/* Wait for all launched workers */
	WaitForParallelWorkersToAttach(pcxt);
}

/*
 * Compute parallel workers
 */
static int
ComputeParallelWorkers(Relation heap, Relation index)
{
	int			parallel_workers;

	/* Make sure it's safe to use parallel workers */
	parallel_workers = plan_create_index_workers(RelationGetRelid(heap), RelationGetRelid(index));
	if (parallel_workers == 0)
		return 0;

	/* Use parallel_workers storage parameter on table if set */
	parallel_workers = RelationGetParallelWorkers(heap, -1);
	if (parallel_workers != -1)
		return Min(parallel_workers, max_parallel_maintenance_workers);

	return max_parallel_maintenance_workers;
}

/*
 * Build graph
 */
static void
BuildGraph(HnswBuildState * buildstate, ForkNumber forkNum)
{
	int			parallel_workers = 0;

	pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_HNSW_PHASE_LOAD);

	/* Calculate parallel workers */
	if (buildstate->heap != NULL)
		parallel_workers = ComputeParallelWorkers(buildstate->heap, buildstate->index);

	/* Attempt to launch parallel worker scan when required */
	if (parallel_workers > 0)
		HnswBeginParallel(buildstate, buildstate->indexInfo->ii_Concurrent, parallel_workers);

	/* Add tuples to graph */
	if (buildstate->heap != NULL)
	{
		if (buildstate->hnswleader)
			buildstate->reltuples = ParallelHeapScan(buildstate);
		else
			buildstate->reltuples = table_index_build_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
														   true, true, BuildCallback, (void *) buildstate, NULL);

		buildstate->indtuples = buildstate->graph->indtuples;
	}

	/* Flush pages */
	if (!buildstate->graph->flushed)
		FlushPages(buildstate);

	/* End parallel build */
	if (buildstate->hnswleader)
		HnswEndParallel(buildstate->hnswleader);
}

/*
 * Build the index
 */
static void
BuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
		   HnswBuildState * buildstate, ForkNumber forkNum)
{
#ifdef HNSW_MEMORY
	SeedRandom(42);
#endif

	InitBuildState(buildstate, heap, index, indexInfo, forkNum);

	BuildGraph(buildstate, forkNum);

	if (RelationNeedsWAL(index) || forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

	FreeBuildState(buildstate);
}

/*
 * Build the index for a logged table
 */
IndexBuildResult *
hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	HnswBuildState buildstate;

	/* Init page dedup set for this build */
	hnsw_io_page_set = palloc0(sizeof(HnswIOPageSet));

	BuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

	hnsw_io_page_set = NULL;

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = buildstate.reltuples;
	result->index_tuples = buildstate.indtuples;

	return result;
}

/*
 * Build the index for an unlogged table
 */
void
hnswbuildempty(Relation index)
{
	IndexInfo  *indexInfo = BuildIndexInfo(index);
	HnswBuildState buildstate;

	/* Init page dedup set for this build */
	hnsw_io_page_set = palloc0(sizeof(HnswIOPageSet));

	BuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);

	hnsw_io_page_set = NULL;
}
