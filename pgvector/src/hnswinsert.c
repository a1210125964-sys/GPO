#include "postgres.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "hnsw.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/datum.h"
#include "utils/memutils.h"

/*
 * Get the insert page
 */
static BlockNumber
GetInsertPage(Relation index)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;
	BlockNumber insertPage;

	int64		io_before = 0;

	if (hnsw_io_track)
		HnswIOCountLayerReadBegin(&io_before);
	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	if (hnsw_io_track)
		HnswIOCountLayerReadEnd(io_before, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	insertPage = metap->insertPage;

	UnlockReleaseBuffer(buf);

	return insertPage;
}

/*
 * Get the centralized NeighborTuple insert page
 */
static BlockNumber
HnswGetNeighborInsertPage(Relation index)
{
	Buffer      buf;
	Page        page;
	HnswMetaPage metap;
	BlockNumber neighborInsertPage;

	int64		io_before = 0;

	if (hnsw_io_track)
		HnswIOCountLayerReadBegin(&io_before);
	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	if (hnsw_io_track)
		HnswIOCountLayerReadEnd(io_before, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	neighborInsertPage = metap->neighborInsertPage;

	UnlockReleaseBuffer(buf);

	return neighborInsertPage;
}

/*
 * Check for a free offset
 */
static bool
HnswFreeOffset(Relation index, Buffer buf, Page page, HnswElement element, Size etupSize, Size ntupSize, Buffer *nbuf, Page *npage, OffsetNumber *freeOffno, OffsetNumber *freeNeighborOffno, BlockNumber *newInsertPage, uint8 *tupleVersion)
{
	OffsetNumber offno;
	OffsetNumber maxoffno = PageGetMaxOffsetNumber(page);

	for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
	{
		ItemId		eitemid = PageGetItemId(page, offno);
		HnswElementTuple etup = (HnswElementTuple) PageGetItem(page, eitemid);

		/* Skip neighbor tuples */
		if (!HnswIsElementTuple(etup))
			continue;

		if (etup->deleted)
		{
			BlockNumber elementPage = BufferGetBlockNumber(buf);
			BlockNumber neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
			OffsetNumber neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
			ItemId		nitemid;
			Size		pageFree;
			Size		npageFree;

			if (!BlockNumberIsValid(*newInsertPage))
				*newInsertPage = elementPage;

			/*
			 * Centralized design: NeighborTuple is always on an independent
			 * centralized page. Remove the same-page branch (original lines 67-71).
			 *
			 * WAL semantics (important):
			 *   This function reads nbuf via ReadBuffer and BufferGetPage (no
			 *   GenericXLogRegisterBuffer). This is a "not yet WAL-registered"
			 *   read. The caller AddElementOnDisk will later register this buffer
			 *   via GenericXLog in its own state.
			 *
			 *   Using BufferGetPage here is safe because this function only reads
			 *   the page (computing free space), not modifying it. When the caller
			 *   later calls GenericXLogRegisterBuffer(state, nbuf, 0), flag 0
			 *   (delta mode) snapshots the current page content as baseline —
			 *   which is identical to what we read here (we hold exclusive lock,
			 *   no other backend can modify it).
			 *
			 *   Note: In the centralized design, AddElementOnDisk's Phase 1 aborts
			 *   its current GenericXLog state after HnswFreeOffset returns, then
			 *   creates a new state in Phase 3 and registers nbuf there. nbuf
			 *   holds BUFFER_LOCK_EXCLUSIVE across the two GenericXLog intervals,
			 *   so flag 0 (delta) is still correct.
			 */
			Assert(neighborPage != elementPage);

			int64		io_before = 0;

			if (hnsw_io_track)
				HnswIOCountLayerReadBegin(&io_before);
			*nbuf = ReadBuffer(index, neighborPage);
			if (hnsw_io_track)
				HnswIOCountLayerReadEnd(io_before, neighborPage);
			LockBuffer(*nbuf, BUFFER_LOCK_EXCLUSIVE);

			*npage = BufferGetPage(*nbuf);

			nitemid = PageGetItemId(*npage, neighborOffno);

			/* Ensure aligned for space check */
			Assert(etupSize == MAXALIGN(etupSize));
			Assert(ntupSize == MAXALIGN(ntupSize));

			/*
			 * Free space calculation (centralized version):
			 * NeighborTuple is always on a different page, compute each
			 * page's free space independently.
			 */
			pageFree = ItemIdGetLength(eitemid) + PageGetExactFreeSpace(page);
			npageFree = ItemIdGetLength(nitemid) + PageGetExactFreeSpace(*npage);

			/* Check for space */
			if (pageFree >= etupSize && npageFree >= ntupSize)
			{
				*freeOffno = offno;
				*freeNeighborOffno = neighborOffno;
				*tupleVersion = etup->version;
				return true;
			}
			else
				UnlockReleaseBuffer(*nbuf);
		}
	}

	return false;
}

/*
 * Add a new page
 */
static void
HnswInsertAppendPage(Relation index, Buffer *nbuf, Page *npage, GenericXLogState *state, Page page, bool building)
{
	/* Add a new page — P_NEW is allocation, not a read */
	LockRelationForExtension(index, ExclusiveLock);
	*nbuf = HnswNewBuffer(index, MAIN_FORKNUM);
	UnlockRelationForExtension(index, ExclusiveLock);

	/* Init new page */
	if (building)
		*npage = BufferGetPage(*nbuf);
	else
		*npage = GenericXLogRegisterBuffer(state, *nbuf, GENERIC_XLOG_FULL_IMAGE);

	HnswInitPage(*nbuf, *npage);

	/* Update previous buffer */
	HnswPageGetOpaque(page)->nextblkno = BufferGetBlockNumber(*nbuf);
}

/*
 * Add element and neighbor to disk — centralized NeighborTuple version
 *
 * ElementTuple → regular page chain (metapage->insertPage tracks)
 * NeighborTuple → centralized page chain (metapage->neighborInsertPage tracks)
 *
 * GenericXLog strategy:
 *   Non-build path: GenericXLogStart once, RegisterBuffer twice (ebuf + nbuf),
 *                  GenericXLogFinish once for atomic commit. Page extensions
 *                  need independent cycles.
 *   Build path:     BufferGetPage + MarkBufferDirty, no GenericXLog.
 */
static void
AddElementOnDisk(Relation index, HnswElement e, int m,
                 BlockNumber insertPage, BlockNumber *updatedInsertPage,
                 BlockNumber neighborInsertPage, BlockNumber *updatedNeighborInsertPage,
                 bool building)
{
	Buffer			ebuf;		/* ElementTuple page buffer */
	Page			epage;		/* ElementTuple page */
	Buffer			nbuf;		/* NeighborTuple page buffer */
	Page			npage;		/* NeighborTuple page */
	GenericXLogState *state = NULL;
	Size			etupSize;
	Size			ntupSize;
	HnswElementTuple etup;
	HnswNeighborTuple ntup;
	BlockNumber		eCurrentPage = insertPage;
	BlockNumber		nCurrentPage = neighborInsertPage;
	OffsetNumber	freeOffno = InvalidOffsetNumber;
	OffsetNumber	freeNeighborOffno = InvalidOffsetNumber;
	BlockNumber		newInsertPage = InvalidBlockNumber;
	BlockNumber		newNeighborInsertPage = InvalidBlockNumber;
	uint8			tupleVersion;
	bool			elementPlaced = false;
	char		   *base = NULL;

	/* Calculate sizes */
	etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(HnswPtrAccess(base, e->value)));
	ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(e->level, m);

	/* Prepare tuples (heap memory, copied onto pages via PageAddItem/PageIndexTupleOverwrite) */
	etup = palloc0(etupSize);
	HnswSetElementTuple(base, etup, e);

	ntup = palloc0(ntupSize);
	HnswSetNeighborTuple(base, ntup, e, m);

	/*
	 * ============================================================
	 * Phase 1: Find/allocate ElementTuple page
	 * ============================================================
	 * Walk the ElementTuple page chain looking for a page with
	 * enough free space. Logic is similar to the original but no
	 * longer concerns itself with NeighborTuple page location.
	 */
	for (;;)
	{
		int64		io_before = 0;

		if (hnsw_io_track)
		if (hnsw_io_track)
			HnswIOCountLayerReadBegin(&io_before);
		ebuf = ReadBuffer(index, eCurrentPage);
		if (hnsw_io_track)
			HnswIOCountLayerReadEnd(io_before, eCurrentPage);
		LockBuffer(ebuf, BUFFER_LOCK_EXCLUSIVE);

		if (building)
		{
			epage = BufferGetPage(ebuf);
		}
		else
		{
			/* Temporary state for page traversal only — final registration in Phase 3 */
			state = GenericXLogStart(index);
			epage = GenericXLogRegisterBuffer(state, ebuf, 0);
		}

		/* Track first page where element can fit */
		if (!BlockNumberIsValid(newInsertPage) && PageGetFreeSpace(epage) >= etupSize)
			newInsertPage = eCurrentPage;

		/* Check for free offset from a deleted element */
		if (HnswFreeOffset(index, ebuf, epage, e, etupSize, ntupSize,
		                   &nbuf, &npage, &freeOffno, &freeNeighborOffno,
		                   &newInsertPage, &tupleVersion))
		{
			/*
			 * Found a deleted element's slot to reuse.
			 * HnswFreeOffset has already:
			 *   - LockBuffer(nbuf) on the NeighborTuple page
			 *   - Verified both pages have enough space
			 * Just abort the current traversal state; re-register in Phase 3/4.
			 */
			if (!building)
				GenericXLogAbort(state);
			state = NULL;	/* will be re-created in Phase 3/4 */

			e->blkno = BufferGetBlockNumber(ebuf);
			e->offno = freeOffno;
			e->neighborPage = BufferGetBlockNumber(nbuf);
			e->neighborOffno = freeNeighborOffno;

			/* Set version to match the deleted tuple being overwritten */
			etup->version = tupleVersion;
			ntup->version = tupleVersion;

			elementPlaced = true;
			break;
		}

		/* Enough space on current page for ElementTuple? */
		if (PageGetFreeSpace(epage) >= etupSize)
		{
			if (!building)
				GenericXLogAbort(state);
			state = NULL;

			/*
			 * GenericXLogAbort releases its internal page reference,
			 * making epage a dangling pointer. Must re-fetch.
			 * ebuf is still pinned and holds exclusive lock, so safe.
			 */
			epage = BufferGetPage(ebuf);

			e->blkno = BufferGetBlockNumber(ebuf);
			e->offno = OffsetNumberNext(PageGetMaxOffsetNumber(epage));

			/* NeighborTuple page not determined yet — Phase 2 handles it */
			break;
		}

		/* Move to next page */
		eCurrentPage = HnswPageGetOpaque(epage)->nextblkno;

		if (BlockNumberIsValid(eCurrentPage))
		{
			if (!building)
				GenericXLogAbort(state);
			UnlockReleaseBuffer(ebuf);
		}
		else
		{
			/* End of page chain — append new ElementTuple page */
			Buffer	newbuf;
			Page	newpage;

			HnswInsertAppendPage(index, &newbuf, &newpage, state, epage, building);

			/* Commit old page */
			if (building)
				MarkBufferDirty(ebuf);
			else
				GenericXLogFinish(state);
			state = NULL;
			UnlockReleaseBuffer(ebuf);

			/* Switch to new page */
			ebuf = newbuf;
			if (building)
			{
				epage = BufferGetPage(ebuf);
			}
			else
			{
				state = GenericXLogStart(index);
				epage = GenericXLogRegisterBuffer(state, ebuf, 0);
				GenericXLogAbort(state);	/* abort temporarily, re-register later */
				state = NULL;
			}

			/*
			 * GenericXLogAbort releases internal page reference, epage
			 * becomes dangling. Must re-fetch. ebuf still pinned & locked.
			 */
			epage = BufferGetPage(ebuf);

			e->blkno = BufferGetBlockNumber(ebuf);
			e->offno = OffsetNumberNext(PageGetMaxOffsetNumber(epage));
			break;
		}
	}

	/*
	 * ============================================================
	 * Phase 2: Find/allocate NeighborTuple page (centralized chain)
	 * ============================================================
	 * If Phase 1's HnswFreeOffset already determined nbuf
	 * (elementPlaced=true), skip this phase. Otherwise walk the
	 * centralized NeighborTuple page chain.
	 */
	if (!elementPlaced)
	{
		if (!BlockNumberIsValid(nCurrentPage))
		{
			/* No NeighborTuple pages yet — create the first one (P_NEW is allocation, not a read) */
			LockRelationForExtension(index, ExclusiveLock);
			nbuf = HnswNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);

			if (building)
				npage = BufferGetPage(nbuf);
			else
			{
				/*
				 * New page must use GENERIC_XLOG_FULL_IMAGE (not 0/delta),
				 * because no prior version exists to compute a delta against.
				 * Same strategy as HnswInsertAppendPage (hnswinsert.c:133).
				 */
				state = GenericXLogStart(index);
				npage = GenericXLogRegisterBuffer(state, nbuf, GENERIC_XLOG_FULL_IMAGE);
				GenericXLogFinish(state);
				state = NULL;
			}

			/*
			 * GenericXLogFinish releases its internal page reference,
			 * making npage dangling. Must re-fetch. nbuf is still
			 * pinned and holds exclusive lock, so safe.
			 */
			npage = BufferGetPage(nbuf);

			HnswInitPage(nbuf, npage);
			if (building)
				MarkBufferDirty(nbuf);

			newNeighborInsertPage = BufferGetBlockNumber(nbuf);
			e->neighborPage = BufferGetBlockNumber(nbuf);
			e->neighborOffno = FirstOffsetNumber;
		}
		else
		{
			/* Walk centralized NeighborTuple page chain for space */
			for (;;)
			{
				int64		io_before = 0;

				if (hnsw_io_track)
					HnswIOCountLayerReadBegin(&io_before);
				nbuf = ReadBuffer(index, nCurrentPage);
				if (hnsw_io_track)
					HnswIOCountLayerReadEnd(io_before, nCurrentPage);
				LockBuffer(nbuf, BUFFER_LOCK_EXCLUSIVE);

				if (building)
					npage = BufferGetPage(nbuf);
				else
				{
					state = GenericXLogStart(index);
					npage = GenericXLogRegisterBuffer(state, nbuf, 0);
				}

				if (!BlockNumberIsValid(newNeighborInsertPage))
					newNeighborInsertPage = nCurrentPage;

				/*
				 * Search for a free NeighborTuple slot (hole left by
				 * vacuum deletion). Scan all items on the page for an
				 * unused ItemId whose length >= ntupSize.
				 */
				{
					OffsetNumber offno;
					OffsetNumber maxoffno = PageGetMaxOffsetNumber(npage);

					for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
					{
						ItemId nitemid = PageGetItemId(npage, offno);

						if (!ItemIdIsUsed(nitemid) &&
							ItemIdGetLength(nitemid) >= ntupSize)
						{
							freeNeighborOffno = offno;
							break;
						}
					}
				}

				/* Enough free space (new slot or free slot)? */
				if (OffsetNumberIsValid(freeNeighborOffno) ||
					PageGetFreeSpace(npage) >= ntupSize)
				{
					if (!building)
						GenericXLogAbort(state);
					state = NULL;

					/*
					 * GenericXLogAbort releases internal page reference,
					 * making npage dangling. Must re-fetch.
					 * nbuf is still pinned and holds exclusive lock.
					 */
					npage = BufferGetPage(nbuf);

					e->neighborPage = BufferGetBlockNumber(nbuf);
					if (OffsetNumberIsValid(freeNeighborOffno))
						e->neighborOffno = freeNeighborOffno;
					else
						e->neighborOffno = OffsetNumberNext(PageGetMaxOffsetNumber(npage));
					break;
				}

				/* Move to next NeighborTuple page */
				nCurrentPage = HnswPageGetOpaque(npage)->nextblkno;

				if (BlockNumberIsValid(nCurrentPage))
				{
					if (!building)
						GenericXLogAbort(state);
					UnlockReleaseBuffer(nbuf);
				}
				else
				{
					/* Append new NeighborTuple page */
					Buffer	newnbuf;
					Page	newnpage;

					HnswInsertAppendPage(index, &newnbuf, &newnpage, state, npage, building);

					if (building)
						MarkBufferDirty(nbuf);
					else
						GenericXLogFinish(state);
					state = NULL;
					UnlockReleaseBuffer(nbuf);

					nbuf = newnbuf;
					if (building)
						npage = BufferGetPage(nbuf);
					else
					{
						state = GenericXLogStart(index);
						npage = GenericXLogRegisterBuffer(state, nbuf, 0);
						GenericXLogAbort(state);
						state = NULL;
					}

					newNeighborInsertPage = BufferGetBlockNumber(nbuf);
					e->neighborPage = BufferGetBlockNumber(nbuf);
					e->neighborOffno = FirstOffsetNumber;
					break;
				}
			}
		}
	}

	ItemPointerSet(&etup->neighbortid, e->neighborPage, e->neighborOffno);

	/* Determine updated insert pages */
	if (!BlockNumberIsValid(newInsertPage))
		newInsertPage = e->blkno;
	if (!BlockNumberIsValid(newNeighborInsertPage))
		newNeighborInsertPage = e->neighborPage;

	/*
	 * ============================================================
	 * Phase 3: GenericXLog dual-page atomic registration & write
	 * ============================================================
	 * At this point ebuf and nbuf are both locked (BUFFER_LOCK_EXCLUSIVE).
	 * Neither buffer is registered in a prior GenericXLog state
	 * (any temporary states from Phase 1/2 have been aborted/finished).
	 *
	 * Register both buffers in a new state — flag=0 (delta) applies to
	 * both, since they haven't been modified since initial ReadBuffer,
	 * so the current content IS the baseline.
	 */
	if (!building)
	{
		Assert(state == NULL);
		state = GenericXLogStart(index);
		epage = GenericXLogRegisterBuffer(state, ebuf, 0);
		npage = GenericXLogRegisterBuffer(state, nbuf, 0);
	}
	else
	{
		epage = BufferGetPage(ebuf);
		npage = BufferGetPage(nbuf);
	}

	/* Write ElementTuple */
	if (OffsetNumberIsValid(freeOffno))
	{
		if (!PageIndexTupleOverwrite(epage, e->offno, (Item) etup, etupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}
	else
	{
		if (PageAddItem(epage, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != e->offno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}

	/* Write NeighborTuple (to centralized page) */
	if (OffsetNumberIsValid(freeNeighborOffno))
	{
		if (!PageIndexTupleOverwrite(npage, e->neighborOffno, (Item) ntup, ntupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}
	else
	{
		if (PageAddItem(npage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != e->neighborOffno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}

	/*
	 * ============================================================
	 * Phase 4: Commit & unlock
	 * ============================================================
	 */
	if (building)
	{
		MarkBufferDirty(ebuf);
		hnsw_io_write_pages++;
		if (HnswIOMarkPageUpdated(BufferGetBlockNumber(ebuf)))
		{
			hnsw_io_write_reason_counts[HNSW_WRITE_ADD_ELEMENT]++;
			hnsw_io_updated_pages++;
		}
		hnsw_io_total_ios++;

		MarkBufferDirty(nbuf);
		hnsw_io_write_pages++;
		if (HnswIOMarkPageUpdated(BufferGetBlockNumber(nbuf)))
		{
			hnsw_io_write_reason_counts[HNSW_WRITE_ADD_ELEMENT]++;
			hnsw_io_updated_pages++;
		}
		hnsw_io_total_ios++;
	}
	else
	{
		/* Atomic commit of both page changes */
		GenericXLogFinish(state);
		hnsw_io_write_pages += 2;
		if (HnswIOMarkPageUpdated(BufferGetBlockNumber(ebuf)))
		{
			hnsw_io_write_reason_counts[HNSW_WRITE_ADD_ELEMENT]++;
			hnsw_io_updated_pages++;
		}
		if (HnswIOMarkPageUpdated(BufferGetBlockNumber(nbuf)))
		{
			hnsw_io_write_reason_counts[HNSW_WRITE_ADD_ELEMENT]++;
			hnsw_io_updated_pages++;
		}
		hnsw_io_total_ios += 2;
	}

	UnlockReleaseBuffer(ebuf);
	UnlockReleaseBuffer(nbuf);

	/* Output updated insert pages */
	if (BlockNumberIsValid(newInsertPage) && newInsertPage != insertPage)
		*updatedInsertPage = newInsertPage;
	if (BlockNumberIsValid(newNeighborInsertPage) && newNeighborInsertPage != neighborInsertPage)
		*updatedNeighborInsertPage = newNeighborInsertPage;

	pfree(etup);
	pfree(ntup);
}

/*
 * Load neighbors
 */
static HnswNeighborArray *
HnswLoadNeighbors(HnswElement element, Relation index, int m, int lm, int lc)
{
	char	   *base = NULL;
	HnswNeighborArray *neighbors = HnswInitNeighborArray(lm, NULL);
	ItemPointerData indextids[HNSW_MAX_M * 2];

	if (!HnswLoadNeighborTids(element, indextids, index, m, lm, lc))
		return neighbors;

	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		HnswElement e;
		HnswCandidate *hc;

		if (!ItemPointerIsValid(indextid))
			break;

		e = HnswInitElementFromBlock(ItemPointerGetBlockNumber(indextid), ItemPointerGetOffsetNumber(indextid));
		hc = &neighbors->items[neighbors->length++];
		HnswPtrStore(base, hc->element, e);
	}

	return neighbors;
}

/*
 * Load elements for insert
 */
static void
LoadElementsForInsert(HnswNeighborArray * neighbors, HnswQuery * q, int *idx, Relation index, HnswSupport * support)
{
	char	   *base = NULL;

	for (int i = 0; i < neighbors->length; i++)
	{
		HnswCandidate *hc = &neighbors->items[i];
		HnswElement element = HnswPtrAccess(base, hc->element);
		double		distance;

		HnswLoadElement(element, &distance, q, index, support, true, NULL);
		hc->distance = distance;

		/* Prune element if being deleted */
		if (element->heaptidsLength == 0)
		{
			*idx = i;
			break;
		}
	}
}

/*
 * Get update index
 */
static int
GetUpdateIndex(HnswElement element, HnswElement newElement, float distance, int m, int lm, int lc, Relation index, HnswSupport * support, MemoryContext updateCtx)
{
	char	   *base = NULL;
	int			idx = -1;
	HnswNeighborArray *neighbors;
	MemoryContext oldCtx = MemoryContextSwitchTo(updateCtx);

	/*
	 * Get latest neighbors since they may have changed. Do not lock yet since
	 * selecting neighbors can take time. Could use optimistic locking to
	 * retry if another update occurs before getting exclusive lock.
	 */
	neighbors = HnswLoadNeighbors(element, index, m, lm, lc);

	/*
	 * Could improve performance for vacuuming by checking neighbors against
	 * list of elements being deleted to find index. It's important to exclude
	 * already deleted elements for this since they can be replaced at any
	 * time.
	 */

	if (neighbors->length < lm)
		idx = -2;
	else
	{
		HnswQuery	q;

		q.value = HnswGetValue(base, element);

		LoadElementsForInsert(neighbors, &q, &idx, index, support);

		if (idx == -1)
			HnswUpdateConnection(base, neighbors, newElement, distance, lm, &idx, index, support);
	}

	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(updateCtx);

	return idx;
}

/*
 * Check if connection already exists
 */
static bool
ConnectionExists(HnswElement e, HnswNeighborTuple ntup, int startIdx, int lm)
{
	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &ntup->indextids[startIdx + i];

		if (!ItemPointerIsValid(indextid))
			break;

		if (ItemPointerGetBlockNumber(indextid) == e->blkno && ItemPointerGetOffsetNumber(indextid) == e->offno)
			return true;
	}

	return false;
}

/*
 * Update neighbor
 */
static void
UpdateNeighborOnDisk(HnswElement element, HnswElement newElement, int idx, int m, int lm, int lc, Relation index, bool checkExisting, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	HnswNeighborTuple ntup;
	int			startIdx;
	OffsetNumber offno = element->neighborOffno;

	/* Register page */
	int64		io_before = 0;

	if (hnsw_io_track)
		HnswIOCountLayerReadBegin(&io_before);
	buf = ReadBuffer(index, element->neighborPage);
	if (hnsw_io_track)
		HnswIOCountLayerReadEnd(io_before, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	/* Get tuple */
	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, offno));

	/* Calculate index for update */
	startIdx = (element->level - lc) * m;

	/* Check for existing connection */
	if (checkExisting && ConnectionExists(newElement, ntup, startIdx, lm))
		idx = -1;
	else if (idx == -2)
	{
		/* Find free offset if still exists */
		/* TODO Retry updating connections if not */
		for (int j = 0; j < lm; j++)
		{
			if (!ItemPointerIsValid(&ntup->indextids[startIdx + j]))
			{
				idx = startIdx + j;
				break;
			}
		}
	}
	else
		idx += startIdx;

	/* Make robust to issues */
	if (idx >= 0 && idx < ntup->count)
	{
		ItemPointer indextid = &ntup->indextids[idx];

		/* Update neighbor on the buffer */
		ItemPointerSet(indextid, newElement->blkno, newElement->offno);

		/* Commit */
		if (building)
		{
			MarkBufferDirty(buf);
			hnsw_io_write_pages++;
			if (HnswIOMarkPageUpdated(BufferGetBlockNumber(buf)))
			{
				hnsw_io_write_reason_counts[HNSW_WRITE_UPDATE_NEIGHBOR]++;
				hnsw_io_updated_pages++;
			}
			hnsw_io_total_ios++;
		}
		else
		{
			GenericXLogFinish(state);
			hnsw_io_write_pages++;
			if (HnswIOMarkPageUpdated(BufferGetBlockNumber(buf)))
			{
				hnsw_io_write_reason_counts[HNSW_WRITE_UPDATE_NEIGHBOR]++;
				hnsw_io_updated_pages++;
			}
			hnsw_io_total_ios++;
		}
	}
	else if (!building)
		GenericXLogAbort(state);

	UnlockReleaseBuffer(buf);
}

/*
 * Update neighbors
 */
void
HnswUpdateNeighborsOnDisk(Relation index, HnswSupport * support, HnswElement e, int m, bool checkExisting, bool building)
{
	char	   *base = NULL;

	/* Use separate memory context to improve performance for larger vectors */
	MemoryContext updateCtx = GenerationContextCreate(CurrentMemoryContext,
													  "Hnsw insert update context",
#if PG_VERSION_NUM >= 150000
													  128 * 1024, 128 * 1024,
#endif
													  128 * 1024);

	/* Defensive reset: clean up stale layer state from prior ERROR */
	hnsw_io_current_layer = -1;

	for (int lc = e->level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);

		hnsw_io_current_layer = lc;

		for (int i = 0; i < neighbors->length; i++)
		{
			HnswCandidate *hc = &neighbors->items[i];
			HnswElement neighborElement = HnswPtrAccess(base, hc->element);
			int			idx;

			idx = GetUpdateIndex(neighborElement, e, hc->distance, m, lm, lc, index, support, updateCtx);

			/* New element was not selected as a neighbor */
			if (idx == -1)
				continue;

			UpdateNeighborOnDisk(neighborElement, e, idx, m, lm, lc, index, checkExisting, building);
		}
	}

	hnsw_io_current_layer = -1;

	MemoryContextDelete(updateCtx);
}

/*
 * Add a heap TID to an existing element
 */
static bool
AddDuplicateOnDisk(Relation index, HnswElement element, HnswElement dup, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	HnswElementTuple etup;
	int			i;

	/* Read page */
	int64		io_before = 0;

	if (hnsw_io_track)
		HnswIOCountLayerReadBegin(&io_before);
	buf = ReadBuffer(index, dup->blkno);
	if (hnsw_io_track)
		HnswIOCountLayerReadEnd(io_before, dup->blkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	/* Find space */
	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, dup->offno));
	for (i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (!ItemPointerIsValid(&etup->heaptids[i]))
			break;
	}

	/* Either being deleted or we lost our chance to another backend */
	if (i == 0 || i == HNSW_HEAPTIDS)
	{
		if (!building)
			GenericXLogAbort(state);
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Add heap TID, modifying the tuple on the page directly */
	etup->heaptids[i] = element->heaptids[0];

	/* Commit */
	if (building)
	{
		MarkBufferDirty(buf);
		hnsw_io_write_pages++;
		if (HnswIOMarkPageUpdated(BufferGetBlockNumber(buf)))
		{
			hnsw_io_write_reason_counts[HNSW_WRITE_ADD_DUPLICATE]++;
			hnsw_io_updated_pages++;
		}
		hnsw_io_total_ios++;
	}
	else
	{
		GenericXLogFinish(state);
		hnsw_io_write_pages++;
		if (HnswIOMarkPageUpdated(BufferGetBlockNumber(buf)))
		{
			hnsw_io_write_reason_counts[HNSW_WRITE_ADD_DUPLICATE]++;
			hnsw_io_updated_pages++;
		}
		hnsw_io_total_ios++;
	}
	UnlockReleaseBuffer(buf);

	return true;
}

/*
 * Find duplicate element
 */
static bool
FindDuplicateOnDisk(Relation index, HnswElement element, bool building)
{
	char	   *base = NULL;
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

		if (AddDuplicateOnDisk(index, element, neighborElement, building))
			return true;
	}

	return false;
}

/*
 * Update graph on disk
 */
static void
UpdateGraphOnDisk(Relation index, HnswSupport * support, HnswElement element, int m, int efConstruction, HnswElement entryPoint, bool building)
{
	BlockNumber newInsertPage = InvalidBlockNumber;
	BlockNumber newNeighborInsertPage = InvalidBlockNumber;
	BlockNumber insertPage = GetInsertPage(index);
	BlockNumber neighborInsertPage = HnswGetNeighborInsertPage(index);

	/* Look for duplicate */
	if (FindDuplicateOnDisk(index, element, building))
		return;

	/* Add element (centralized: ElementTuple + NeighborTuple on separate page chains) */
	AddElementOnDisk(index, element, m,
	                 insertPage, &newInsertPage,
	                 neighborInsertPage, &newNeighborInsertPage,
	                 building);

	/* Update metapage if either insert page changed */
	if (BlockNumberIsValid(newInsertPage) || BlockNumberIsValid(newNeighborInsertPage))
		HnswUpdateMetaPage(index, 0, NULL,
		                   BlockNumberIsValid(newInsertPage) ? newInsertPage : InvalidBlockNumber,
		                   BlockNumberIsValid(newNeighborInsertPage) ? newNeighborInsertPage : InvalidBlockNumber,
		                   MAIN_FORKNUM, building,
		                   HNSW_WRITE_UPDATE_META_INSERT_PAGE);

	/* Update neighbors (reverse connections — UpdateNeighborOnDisk needs no changes) */
	HnswUpdateNeighborsOnDisk(index, support, element, m, false, building);

	/* Update entry point if needed */
	if (entryPoint == NULL || element->level > entryPoint->level)
		HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_GREATER, element,
		                   InvalidBlockNumber, InvalidBlockNumber,
		                   MAIN_FORKNUM, building,
		                   HNSW_WRITE_UPDATE_META_ENTRY_POINT);
}

/*
 * Insert a tuple into the index
 */
bool
HnswInsertTupleOnDisk(Relation index, HnswSupport * support, Datum value, ItemPointer heaptid, bool building)
{
	HnswElement entryPoint;
	HnswElement element;
	int			m;
	int			efConstruction = HnswGetEfConstruction(index);
	LOCKMODE	lockmode = ShareLock;
	char	   *base = NULL;

	/*
	 * Get a shared lock. This allows vacuum to ensure no in-flight inserts
	 * before repairing graph. Use a page lock so it does not interfere with
	 * buffer lock (or reads when vacuuming).
	 */
	LockPage(index, HNSW_UPDATE_LOCK, lockmode);

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	/* Create an element */
	element = HnswInitElement(base, heaptid, m, HnswGetMl(m), HnswGetMaxLevel(m), NULL);
	HnswPtrStore(base, element->value, DatumGetPointer(value));

	/* Prevent concurrent inserts when likely updating entry point */
	if (entryPoint == NULL || element->level > entryPoint->level)
	{
		/* Release shared lock */
		UnlockPage(index, HNSW_UPDATE_LOCK, lockmode);

		/* Get exclusive lock */
		lockmode = ExclusiveLock;
		LockPage(index, HNSW_UPDATE_LOCK, lockmode);

		/* Get latest entry point after lock is acquired */
		entryPoint = HnswGetEntryPoint(index);
	}

	/* Find neighbors for element */
	HnswFindElementNeighbors(base, element, entryPoint, index, support, m, efConstruction, false);

	/* Update graph on disk */
	UpdateGraphOnDisk(index, support, element, m, efConstruction, entryPoint, building);

	/* Release lock */
	UnlockPage(index, HNSW_UPDATE_LOCK, lockmode);

	return true;
}

/*
 * Insert a tuple into the index
 */
static void
HnswInsertTuple(Relation index, Datum *values, bool *isnull, ItemPointer heaptid)
{
	Datum		value;
	const		HnswTypeInfo *typeInfo = HnswGetTypeInfo(index);
	HnswSupport support;

	HnswInitSupport(&support, index);

	/* Form index value */
	if (!HnswFormIndexValue(&value, values, isnull, typeInfo, &support))
		return;

	HnswInsertTupleOnDisk(index, &support, value, heaptid, false);
}

/*
 * Insert a tuple into the index
 */
bool
hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
		   Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
		   ,bool indexUnchanged
#endif
		   ,IndexInfo *indexInfo
)
{
	MemoryContext oldCtx;
	MemoryContext insertCtx;

	/* Skip nulls */
	if (isnull[0])
		return false;

	/* Init page dedup set for this statement (persists across per-row calls) */
	if (indexInfo->ii_AmCache == NULL)
		indexInfo->ii_AmCache = palloc0(sizeof(HnswIOPageSet));
	hnsw_io_page_set = (HnswIOPageSet *) indexInfo->ii_AmCache;

	/* Create memory context */
	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "Hnsw insert temporary context",
									  ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(insertCtx);

	/* Defensive reset: clean up stale state from prior ERROR */
	hnsw_io_track = false;
	hnsw_io_current_layer = -1;

	/* Insert tuple */
	hnsw_io_track = true;
	HnswInsertTuple(index, values, isnull, heap_tid);
	hnsw_io_track = false;

	/* Delete memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(insertCtx);

	return false;
}
