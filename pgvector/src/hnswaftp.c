/* ----------------------------------------------------------------
 * hnswaftp.c -- AFTP (Access Frequency-driven Tuple Packing)
 *
 * Phase 1: Trace Collection
 *
 * Build-time optimization that reorders tuple placement within pages
 * to improve buffer hit rate during HNSW search.
 *
 * Phase 1 collects access traces: for each sample query, we run a
 * modified HNSW search (in-memory) that records every NeighborTuple
 * and ElementTuple access in a trace entry array. These traces feed
 * Phase 2 (affinity matrix construction).
 *
 * Algorithm (from plan Section 3.2, Algorithm 1):
 *   1. Assign sequential tuple IDs to all elements/neighbors
 *   2. Sample S queries via reservoir sampling
 *   3. For each sample, run multi-layer trace search
 *   4. Check convergence (stub for now, returns true)
 * ----------------------------------------------------------------
 */

#include "postgres.h"

#include <math.h>
#include <float.h>

#include "hnsw.h"
#include "lib/pairingheap.h"
#include "portability/instr_time.h"

/* GUC variables -- defined here, declared extern in hnswaftp.h */
bool	aftp_enable = false;
int		aftp_sample_queries = 200;
int		aftp_ef_search = 64;
int		aftp_window = 100;
int		aftp_top_affinity = 20;
int		aftp_refine_iterations = 5;
double	aftp_decay_lambda = 0;		/* 0 = auto (W/3) */
int		aftp_seed_softmax = 10;
int		aftp_max_memory = 0;		/* 0 = auto (maintenance_work_mem * 0.5) */

/* Return elapsed wall-clock time since start in milliseconds. */
static double
AftpElapsedMilliseconds(instr_time start)
{
	instr_time	end;

	INSTR_TIME_SET_CURRENT(end);
	INSTR_TIME_SUBTRACT(end, start);

	return INSTR_TIME_GET_MILLISEC(end);
}

/* ----------------------------------------------------------------
 * Comparison functions for pairing heaps (replicated from
 * hnswutils.c, where they are static).
 *
 * CompareNearestCandidates -- used by C heap (min-heap by distance)
 *   Return -1 if a->distance > b->distance  =>  a comes first
 * CompareFurthestCandidates -- used by W heap (max-heap by distance,
 *   limited to ef entries)
 * ---------------------------------------------------------------- */
static int
AftpCompareNearest(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(c_node, a)->distance >
		HnswGetSearchCandidateConst(c_node, b)->distance)
		return -1;

	return 0;
}

static int
AftpCompareFurthest(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance <
		HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance >
		HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	return 0;
}

/* ----------------------------------------------------------------
 * Distance functions (replicated from hnswutils.c where they are
 * static / static inline).
 * ---------------------------------------------------------------- */

static inline double
AftpGetDistance(Datum a, Datum b, HnswSupport *support)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo,
											support->collation, a, b));
}

static double
AftpElementDistance(char *base, HnswElement element, HnswQuery *q,
					HnswSupport *support)
{
	Datum		value = HnswGetValue(base, element);

	return AftpGetDistance(q->value, value, support);
}

/* ----------------------------------------------------------------
 * Candidate allocation (replicated from hnswutils.c).
 * ---------------------------------------------------------------- */

static HnswSearchCandidate *
AftpInitSearchCandidate(char *base, HnswElement element, double distance)
{
	HnswSearchCandidate *sc = palloc(sizeof(HnswSearchCandidate));

	HnswPtrStore(base, sc->element, element);
	sc->distance = distance;
	return sc;
}

/* ----------------------------------------------------------------
 * Visited-hash helpers (replicated from hnswutils.c).
 *
 * For in-memory builds (index == NULL), we use a pointer or offset
 * hash table keyed by element address.
 * ---------------------------------------------------------------- */

static inline void
AftpInitVisited(char *base, visited_hash *v, bool inMemory, int ef, int m)
{
	if (!inMemory)
		v->tids = tidhash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else if (base != NULL)
		v->offsets = offsethash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else
		v->pointers = pointerhash_create(CurrentMemoryContext, ef * m * 2, NULL);
}

static inline void
AftpAddToVisited(char *base, visited_hash *v, HnswElementPtr elementPtr,
				 bool inMemory, bool *found)
{
	if (!inMemory)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);
		ItemPointerData indextid;

		ItemPointerSet(&indextid, element->blkno, element->offno);
		tidhash_insert(v->tids, indextid, found);
	}
	else if (base != NULL)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		offsethash_insert_hash(v->offsets, HnswPtrOffset(elementPtr),
							   element->hash, found);
	}
	else
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		pointerhash_insert_hash(v->pointers,
								(uintptr_t) HnswPtrPointer(elementPtr),
								element->hash, found);
	}
}

/* ----------------------------------------------------------------
 * Count element towards ef (replicated from hnswutils.c).
 * When skipElement is NULL (trace mode, no insertion), always true.
 * ---------------------------------------------------------------- */

static inline bool
AftpCountElement(HnswElement skipElement, HnswElement e)
{
	if (skipElement == NULL)
		return true;

	pg_memory_barrier();
	Assert(e);
	return e->heaptidsLength != 0;
}

/* ----------------------------------------------------------------
 * Load unvisited neighbors from memory (replicated from hnswutils.c).
 *
 * Copies the neighborhood into a local buffer under the element lock,
 * then adds each neighbor to the visited set and returns unvisited
 * elements.
 * ---------------------------------------------------------------- */

static void
AftpLoadUnvisitedFromMemory(char *base, HnswElement element,
							HnswUnvisited *unvisited, int *unvisitedLength,
							visited_hash *v, int lc,
							HnswNeighborArray *localNeighborhood,
							Size neighborhoodSize)
{
	HnswNeighborArray *neighborhood = HnswGetNeighbors(base, element, lc);

	/* Copy neighborhood to local memory under the element lock */
	LWLockAcquire(&element->lock, LW_SHARED);
	memcpy(localNeighborhood, neighborhood, neighborhoodSize);
	LWLockRelease(&element->lock);

	*unvisitedLength = 0;

	for (int i = 0; i < localNeighborhood->length; i++)
	{
		HnswCandidate *hc = &localNeighborhood->items[i];
		bool		found;

		AftpAddToVisited(base, v, hc->element, true, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].element = HnswPtrAccess(base, hc->element);
	}
}

/* ----------------------------------------------------------------
 * AftpAssignTupleIds
 *
 * Walk the graph->head linked list and assign sequential IDs:
 *   - element->aftp_tuple_id  (ElementTuple ID)
 *   - element->aftp_ntuple_id (NeighborTuple ID)
 *
 * Output parameters:
 *   *n_elements  = total ElementTuple count
 *   *n_neighbors = total NeighborTuple count
 * ---------------------------------------------------------------- */

void
AftpAssignTupleIds(HnswBuildState *buildstate, int *n_elements,
				   int *n_neighbors)
{
	char	   *base = buildstate->hnswarea;
	HnswElement element = HnswPtrAccess(base, buildstate->graph->head);
	int			eid = 0;
	int			nid = 0;

	while (element != NULL)
	{
		element->aftp_tuple_id = eid++;
		element->aftp_ntuple_id = nid++;
		element = HnswPtrAccess(base, element->next);
	}

	*n_elements = eid;
	*n_neighbors = nid;
}

/* ----------------------------------------------------------------
 * AftpTraceSearchLayer (static)
 *
 * Modified version of HnswSearchLayer that:
 *   - Always runs in memory mode (index == NULL, inMemory == true)
 *   - Records each access as an AftpTraceEntry:
 *       * When popping a candidate c (NeighborTuple read):
 *           (c->aftp_ntuple_id, AFTP_TUPLE_NEIGHBOR, lc)
 *       * When computing distance for each neighbor e (ElementTuple read):
 *           (e.aftp_tuple_id, AFTP_TUPLE_ELEMENT, lc)
 *   - Returns the result list W for use as entry points in the
 *     next (lower) layer.
 *
 * trace may be NULL (for convergence-check internal use).  When
 * non-NULL, entries are dynamically appended; the buffer is
 * doubled whenever capacity is exceeded.
 * ---------------------------------------------------------------- */

static List *
AftpTraceSearchLayer(char *base, HnswQuery *q, List *ep, int ef, int lc,
					 HnswSupport *support, int m, AftpQueryTrace *trace)
{
	List	   *w = NIL;
	pairingheap *C = pairingheap_allocate(AftpCompareNearest, NULL);
	pairingheap *W = pairingheap_allocate(AftpCompareFurthest, NULL);
	int			wlen = 0;
	visited_hash vh;
	visited_hash *v = &vh;
	ListCell   *lc2;
	HnswNeighborArray *localNeighborhood = NULL;
	Size		neighborhoodSize = 0;
	int			lm = HnswGetLayerM(m, lc);
	HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));
	int			unvisitedLength;
	bool		inMemory = true;

	/* Initialise visited hash */
	AftpInitVisited(base, v, inMemory, ef, m);

	/* Allocate local buffer for neighbor arrays */
	neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
	localNeighborhood = palloc(neighborhoodSize);

	/* Add each entry point to v, C, and W */
	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		bool		found;

		AftpAddToVisited(base, v, sc->element, inMemory, &found);

		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);

		if (AftpCountElement(NULL, HnswPtrAccess(base, sc->element)))
			wlen++;
	}

	/* Main search loop with trace recording */
	while (!pairingheap_is_empty(C))
	{
		HnswSearchCandidate *c;

		c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));

		{
			HnswSearchCandidate *f;

			f = HnswGetSearchCandidate(w_node, pairingheap_first(W));

			if (c->distance > f->distance)
				break;
		}

		{
			HnswElement cElement = HnswPtrAccess(base, c->element);

			/*
			 * RECORD TRACE: NeighborTuple access
			 * Popping c means we are about to read c's neighbor list
			 * from its NeighborTuple.
			 */
			if (trace != NULL)
			{
				if (trace->nentries >= trace->max_entries)
				{
					trace->max_entries *= 2;
					trace->entries = repalloc(trace->entries,
											  trace->max_entries * sizeof(AftpTraceEntry));
				}
				trace->entries[trace->nentries].tuple_id = cElement->aftp_ntuple_id;
				trace->entries[trace->nentries].type = AFTP_TUPLE_NEIGHBOR;
				trace->entries[trace->nentries].layer = (uint8) lc;
				trace->nentries++;
			}

			/* Load unvisited neighbors */
			AftpLoadUnvisitedFromMemory(base, cElement, unvisited,
										&unvisitedLength, v, lc,
										localNeighborhood, neighborhoodSize);

			for (int i = 0; i < unvisitedLength; i++)
			{
				HnswElement eElement;
				HnswSearchCandidate *e;
				double		eDistance;
				bool		alwaysAdd = wlen < ef;

				{
					HnswSearchCandidate *f;

					f = HnswGetSearchCandidate(w_node, pairingheap_first(W));

					/* In-memory distance computation */
					eElement = unvisited[i].element;
					eDistance = AftpElementDistance(base, eElement, q, support);

					/*
					 * RECORD TRACE: ElementTuple access
					 * Computing distance for e requires reading
					 * e's ElementTuple (its vector value).
					 */
					if (trace != NULL)
					{
						if (trace->nentries >= trace->max_entries)
						{
							trace->max_entries *= 2;
							trace->entries = repalloc(trace->entries,
													  trace->max_entries * sizeof(AftpTraceEntry));
						}
						trace->entries[trace->nentries].tuple_id = eElement->aftp_tuple_id;
						trace->entries[trace->nentries].type = AFTP_TUPLE_ELEMENT;
						trace->entries[trace->nentries].layer = (uint8) lc;
						trace->nentries++;
					}

					if (eElement == NULL || !(eDistance < f->distance || alwaysAdd))
						continue;
				}

				/* Make robust to issues */
				if (eElement->level < lc)
					continue;

				/* Create a new candidate */
				e = AftpInitSearchCandidate(base, eElement, eDistance);
				pairingheap_add(C, &e->c_node);
				pairingheap_add(W, &e->w_node);

				if (AftpCountElement(NULL, eElement))
				{
					wlen++;

					if (wlen > ef)
					{
						/* discarded: not tracked for trace collection */
						pairingheap_remove_first(W);
					}
				}
			}
		}
	}

	/* Collect W into a List for use as entry points in next layer */
	while (!pairingheap_is_empty(W))
	{
		HnswSearchCandidate *sc;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));
		w = lappend(w, sc);
	}

	return w;
}

/* ----------------------------------------------------------------
 * Reservoir sampling
 *
 * Select S random elements uniformly from the linked list starting
 * at 'head'.  Uses standard Algorithm R (reservoir sampling).
 * ---------------------------------------------------------------- */

static HnswElement *
AftpReservoirSample(char *base, HnswElement head, int n_elements, int S,
					int *out_n)
{
	HnswElement *samples;
	HnswElement element;
	int			i = 0;

	if (S >= n_elements)
		S = n_elements;

	samples = palloc(S * sizeof(HnswElement));

	/* Fill reservoir with first S elements */
	element = head;
	for (i = 0; i < S && element != NULL; i++)
	{
		samples[i] = element;
		element = HnswPtrAccess(base, element->next);
	}

	/* Replace reservoir entries with decreasing probability */
	while (element != NULL)
	{
		int			j = (int) (RandomDouble() * (i + 1));

		if (j < S)
			samples[j] = element;
		i++;
		element = HnswPtrAccess(base, element->next);
	}

	*out_n = S;
	return samples;
}

/* ----------------------------------------------------------------
 * AftpCollectTraces
 *
 * Main Phase 1 entry point.  Steps:
 *   1. Assign sequential IDs to all elements
 *   2. Select S = config->sample_queries random elements via
 *      reservoir sampling
 *   3. For each sample:
 *      a. Build HnswQuery from the element's vector
 *      b. For each layer from entryLevel down to 0, run
 *         AftpTraceSearchLayer, recording all NeighborTuple
 *         and ElementTuple accesses.
 *      c. Entry points for the next layer come from the search
 *         result W.
 *   4. Return AftpTraceState containing all traces.
 *
 * Allocations are done with palloc/palloc0/repalloc, relying on
 * the caller's MemoryContext (typically buildstate->aftp_context).
 * ---------------------------------------------------------------- */

AftpTraceState *
AftpCollectTraces(HnswBuildState *buildstate, AftpConfig *config)
{
	char	   *base = buildstate->hnswarea;
	int			n_elements;
	int			n_neighbors;
	int			S;
	AftpTraceState *state;
	HnswElement *samples;
	int			n_samples;

	/* Step 1: Assign sequential tuple IDs */
	AftpAssignTupleIds(buildstate, &n_elements, &n_neighbors);

	elog(INFO, "AFTP Phase 1: assigned %d ElementTuple IDs, %d NeighborTuple IDs",
		 n_elements, n_neighbors);

	/* Sanity check: need at least one element */
	if (n_elements == 0)
	{
		elog(WARNING, "AFTP Phase 1: no elements in graph, skipping trace collection");
		return NULL;
	}

	/* Step 2: Determine sample count S */
	S = config->sample_queries;
	if (S <= 0)
	{
		elog(INFO, "AFTP Phase 1: sample_queries=%d, skipping trace collection", S);
		return NULL;
	}

	if (S > n_elements)
	{
		elog(NOTICE, "AFTP Phase 1: sample_queries=%d > n_elements=%d, "
			 "using n_elements", S, n_elements);
		S = n_elements;
	}

	/* Reservoir-sample S elements */
	samples = AftpReservoirSample(base,
								  HnswPtrAccess(base, buildstate->graph->head),
								  n_elements, S, &n_samples);

	/* Allocate trace state */
	state = palloc0(sizeof(AftpTraceState));
	state->n_queries = S;
	state->n_completed = 0;
	state->traces = palloc0(S * sizeof(AftpQueryTrace));
	state->n_elements = n_elements;
	state->n_neighbors = n_neighbors;

	/* Step 3: Run trace search for each sample query */
	for (int i = 0; i < n_samples; i++)
	{
		HnswElement sampleElement = samples[i];
		HnswElement entryElement;
		int			entryLevel;
		HnswQuery	q;
		AftpQueryTrace *trace;
		List	   *ep;

		/* Build query from the sample element's vector value */
		q.value = HnswGetValue(base, sampleElement);

		/* Initialise trace buffer (start with 4096 entries, double as needed) */
		trace = &state->traces[state->n_completed];
		trace->nentries = 0;
		trace->max_entries = 4096;
		trace->entries = palloc(trace->max_entries * sizeof(AftpTraceEntry));

		/* Verify entry point exists */
		entryElement = HnswPtrAccess(base, buildstate->graph->entryPoint);
		if (entryElement == NULL)
		{
			elog(WARNING, "AFTP Phase 1: entry point is NULL, skipping sample %d",
				 i);
			pfree(trace->entries);
			trace->entries = NULL;
			continue;
		}

		/* Determine search start level from the entry point */
		entryLevel = entryElement->level;

		/* Create initial entry point candidate */
		ep = list_make1(HnswEntryCandidate(base, entryElement, &q, NULL,
										   &buildstate->support, true));

		/* Multi-layer trace search: entryLevel down to 0 */
		for (int lc = entryLevel; lc >= 0; lc--)
		{
			List	   *w;
			int			ef_search;

			ef_search = (config->ef_search > 0) ? config->ef_search : 64;

			w = AftpTraceSearchLayer(base, &q, ep, ef_search, lc,
									 &buildstate->support, buildstate->m,
									 trace);

			/* Result W becomes entry points for the next (lower) layer */
			ep = w;
		}

		state->n_completed++;
	}

	pfree(samples);

	elog(INFO, "AFTP Phase 1: collected traces for %d / %d sample queries",
		 state->n_completed, S);

	return state;
}

/* ----------------------------------------------------------------
 * AftpCheckTraceConvergence
 *
 * Split traces into first and second halves, compute top-K affinity
 * for each, and check Jaccard similarity >= 0.95.
 *
 * CURRENTLY A STUB: logs a message and returns true.
 * The full implementation will be provided in a later task when
 * the affinity matrix computation (Phase 2) is also available.
 * ---------------------------------------------------------------- */

bool
AftpCheckTraceConvergence(AftpQueryTrace *traces, int n_queries,
						  int n_tuples, int top_affinity)
{
	/*
	 * Full convergence check is deferred to a later task.
	 * The algorithm:
	 *   1. Split traces into first half and second half.
	 *   2. For each half, compute top-K_affinity affinity lists using
	 *      simple co-visit counting (no decay weighting).
	 *   3. Compute average Jaccard similarity between the two lists.
	 *   4. Return true if >= 0.95.
	 */
	elog(INFO, "AFTP Phase 1: convergence check skipped (stub) -- "
		 "returning true");
	return true;
}

/* ----------------------------------------------------------------
 * AftpFreeTraceState
 *
 * Free all dynamically allocated memory in the trace state.
 * ---------------------------------------------------------------- */

void
AftpFreeTraceState(AftpTraceState *state)
{
	if (state == NULL)
		return;

	for (int i = 0; i < state->n_completed; i++)
	{
		if (state->traces[i].entries != NULL)
			pfree(state->traces[i].entries);
	}

	if (state->traces != NULL)
		pfree(state->traces);

	pfree(state);
}

/* ==================================================================
 * Phase 2: Affinity Matrix Construction
 *
 * Builds a sparse affinity matrix from collected query traces.
 * Uses time-decayed co-visit counts within a sliding window.
 *
 * Algorithm (from plan Section 3.3):
 *   1. Process each query trace, accumulating co-visit affinities
 *      in a hash table keyed by (min_id, max_id) pair.
 *   2. For each tuple, extract top-K affinities into a sparse row.
 *   3. Free the accumulator hash table.
 *
 * Implementation: Two-pass approach using PostgreSQL's simplehash.h.
 *   - Pass 1: Single hash table keyed by (min_id << 32 | max_id)
 *     accumulates time-decayed co-visit scores.
 *   - Pass 2: Scan the hash once, maintaining a bounded top-K min-heap
 *     for each tuple while computing point weights from every pair.
 *     Sort only the retained entries into final row order.
 * ================================================================== */

/*
 * Hash entry for pair-keyed affinity accumulation.
 * Uses a 64-bit composite key: (min_id << 32) | max_id, where
 * min_id < max_id ensures symmetry.
 */
typedef struct AftpPairEntry
{
	uint64		pair_key;		/* (min_id << 32) | max_id */
	float		affinity;		/* accumulated time-decayed co-visit score */
	char		status;			/* SH_STATUS marker for simplehash */
}			AftpPairEntry;

/*
 * Splitmix64-based hash function for 64-bit pair keys.
 *
 * The pair keys are (min_id << 32) | max_id where HNSW sequential
 * traversal creates correlated (i, i+d) pairs.  XOR alone collapses
 * to min_id ^ max_id which is highly non-uniform.  Splitmix64
 * avalanches all 64 bits uniformly into the 32-bit output.
 */
static inline uint32
hash_aftp_pair(uint64 key)
{
	key ^= key >> 33;
	key *= UINT64_C(0xff51afd7ed558ccd);
	key ^= key >> 33;
	key *= UINT64_C(0xc4ceb9fe1a85ec53);
	key ^= key >> 33;
	return (uint32) key;
}

#define SH_PREFIX aftppair
#define SH_ELEMENT_TYPE AftpPairEntry
#define SH_KEY_TYPE uint64
#define SH_KEY pair_key
#define SH_HASH_KEY(tb, key) hash_aftp_pair(key)
#define SH_EQUAL(tb, a, b) ((a) == (b))
#define SH_SCOPE static inline
#define SH_DECLARE
#define SH_DEFINE
#include "lib/simplehash.h"

/*
 * Compare affinity entries by retention quality.
 *
 * Higher affinity is better.  Equal affinities are ordered by tuple ID so
 * that both heap eviction and final output are deterministic.  The old
 * affinity-only qsort left equal-affinity entries unordered; choosing the
 * lower tuple ID merely canonicalizes that otherwise equivalent tie.
 */
static inline int
AftpCompareAffinityQuality(const AftpAffinityEntry *a,
						   const AftpAffinityEntry *b)
{
	if (a->affinity > b->affinity)
		return 1;
	if (a->affinity < b->affinity)
		return -1;
	if (a->tuple_id < b->tuple_id)
		return 1;
	if (a->tuple_id > b->tuple_id)
		return -1;
	return 0;
}

/* Comparison function for final affinity-descending row order. */
static int
compare_affinity_desc(const void *a, const void *b)
{
	const AftpAffinityEntry *ea = (const AftpAffinityEntry *) a;
	const AftpAffinityEntry *eb = (const AftpAffinityEntry *) b;
	int			quality = AftpCompareAffinityQuality(ea, eb);

	return -quality;
}

/* Exchange two entries in a row's bounded min-heap. */
static inline void
AftpSwapAffinityEntries(AftpAffinityEntry *a, AftpAffinityEntry *b)
{
	AftpAffinityEntry tmp = *a;

	*a = *b;
	*b = tmp;
}

/* Restore the min-heap invariant after appending an entry. */
static void
AftpAffinityHeapSiftUp(AftpAffinityEntry *entries, uint32 pos)
{
	while (pos > 0)
	{
		uint32		parent = (pos - 1) / 2;

		if (AftpCompareAffinityQuality(&entries[pos], &entries[parent]) >= 0)
			break;

		AftpSwapAffinityEntries(&entries[pos], &entries[parent]);
		pos = parent;
	}
}

/* Restore the min-heap invariant after replacing its root. */
static void
AftpAffinityHeapSiftDown(AftpAffinityEntry *entries, uint32 size)
{
	uint32		pos = 0;

	for (;;)
	{
		uint32		left;
		uint32		right;
		uint32		worst;

		/* Avoid overflow in 2 * pos + 1 by testing against size / 2. */
		if (pos >= size / 2)
			break;

		left = pos * 2 + 1;
		right = left + 1;
		worst = left;

		if (right < size &&
			AftpCompareAffinityQuality(&entries[right], &entries[left]) < 0)
			worst = right;

		if (AftpCompareAffinityQuality(&entries[worst], &entries[pos]) >= 0)
			break;

		AftpSwapAffinityEntries(&entries[pos], &entries[worst]);
		pos = worst;
	}
}

/*
 * Offer one entry to a row's bounded top-K min-heap.  The root is the least
 * desirable retained entry, making eviction O(log K).
 */
static void
AftpAffinityHeapOffer(AftpAffinityRow *row, uint32 capacity,
					  uint32 tuple_id, float affinity)
{
	AftpAffinityEntry candidate;

	if (capacity == 0)
		return;

	candidate.tuple_id = tuple_id;
	candidate.affinity = affinity;

	if (row->entries == NULL)
		row->entries = (AftpAffinityEntry *)
			palloc((Size) capacity * sizeof(AftpAffinityEntry));

	if (row->n_entries < capacity)
	{
		uint32		pos = row->n_entries++;

		row->entries[pos] = candidate;
		AftpAffinityHeapSiftUp(row->entries, pos);
	}
	else if (AftpCompareAffinityQuality(&candidate, &row->entries[0]) > 0)
	{
		row->entries[0] = candidate;
		AftpAffinityHeapSiftDown(row->entries, row->n_entries);
	}
}

/* ----------------------------------------------------------------
 * AftpBuildAffinityMatrix
 *
 * Build a sparse affinity matrix from collected query traces.
 *
 * For each trace, within window W, compute exp(-delta / lambda)
 * weighted co-visit scores, accumulate in a hash table, then
 * extract top-K_affinity per tuple into the result matrix.
 *
 * Parameters:
 *   traces       - Collected trace state from Phase 1
 *   n_tuples     - Total number of tuples (elements or neighbors)
 *   top_affinity - K: max non-zero entries per row
 *   window       - W: co-visit window in trace positions
 *   decay_lambda - lambda: time-decay scaling (0 = auto = W/3)
 *   target_type  - AFTP_TUPLE_ELEMENT or AFTP_TUPLE_NEIGHBOR
 *
 * Returns:
 *   Allocated AftpAffinityMatrix, or NULL on invalid input.
 * ---------------------------------------------------------------- */
AftpAffinityMatrix *
AftpBuildAffinityMatrix(AftpTraceState *traces, int n_tuples,
						int top_affinity, int window, double decay_lambda,
						uint8 target_type)
{
	AftpAffinityMatrix *A;
	aftppair_hash *pairhash;
	uint64		estimated_pairs;
	int			effective_window;
	float		effective_lambda;
	uint32		heap_capacity;
	uint32		hash_members;

	/* ---- Validate inputs ---- */
	if (traces == NULL || traces->n_completed <= 0 || n_tuples <= 0)
	{
		elog(WARNING, "AFTP Phase 2: invalid input (traces=%p, n_tuples=%d)",
			 (void *) traces, n_tuples);
		return NULL;
	}

	/*
	 * Cross-validate n_tuples against trace metadata to prevent
	 * out-of-bounds array access when tuple_ids are used as indices.
	 */
	if (target_type == AFTP_TUPLE_ELEMENT && n_tuples != traces->n_elements)
		elog(ERROR, "AFTP Phase 2: n_tuples=%d does not match "
			 "traces->n_elements=%d for element type",
			 n_tuples, traces->n_elements);
	if (target_type == AFTP_TUPLE_NEIGHBOR && n_tuples != traces->n_neighbors)
		elog(ERROR, "AFTP Phase 2: n_tuples=%d does not match "
			 "traces->n_neighbors=%d for neighbor type",
			 n_tuples, traces->n_neighbors);

	if (top_affinity <= 0)
	{
		elog(NOTICE, "AFTP Phase 2: invalid top_affinity=%d, using default 20",
			 top_affinity);
		top_affinity = 20;
	}

	/*
	 * A tuple has at most n_tuples - 1 distinct partners.  Capping the
	 * physical heap at that value preserves the requested top-K semantics and
	 * prevents oversized allocations when K is larger than the data set.
	 */
	heap_capacity = (uint32) Min(top_affinity, n_tuples - 1);

	if ((Size) n_tuples > MaxAllocSize / sizeof(AftpAffinityRow) ||
		(Size) n_tuples > MaxAllocSize / sizeof(float) ||
		(Size) heap_capacity > MaxAllocSize / sizeof(AftpAffinityEntry))
		elog(ERROR, "AFTP Phase 2: affinity matrix allocation is too large "
			 "(n_tuples=%d, K=%d)", n_tuples, top_affinity);

	effective_window = (window > 0) ? window : 100;
	effective_lambda = (decay_lambda > 0.0) ? (float) decay_lambda :
		(float) effective_window / 3.0f;
	/* Guard against zero or denormal lambda → division by zero in expf(-delta/lambda) */
	if (effective_lambda < FLT_MIN)
		effective_lambda = (float) effective_window / 3.0f;

	elog(INFO, "AFTP Phase 2: n_tuples=%d, K=%d, W=%d, lambda=%.2f, type=%s",
		 n_tuples, top_affinity, effective_window, (double) effective_lambda,
		 target_type == AFTP_TUPLE_ELEMENT ? "element" : "neighbor");

	/* ---- Step 1: Allocate result matrix ---- */
	A = (AftpAffinityMatrix *) palloc0(sizeof(AftpAffinityMatrix));
	A->n_tuples = (uint32) n_tuples;
	A->max_entries = (uint32) top_affinity;
	A->rows = (AftpAffinityRow *) palloc0(n_tuples * sizeof(AftpAffinityRow));
	A->point_weights = (float *) palloc0(n_tuples * sizeof(float));

	/* ---- Step 2: Accumulate co-visit pairs in hash table ---- */

	/*
	 * Estimate initial hash size.  Expected unique pairs is roughly
	 * n_tuples * avg_degree / 2.  Start with a generous estimate to
	 * minimize resizing.
	 */
	estimated_pairs = Max(UINT64_C(1024),
						  (uint64) n_tuples * (uint64) top_affinity / 2);
	if (estimated_pairs > PG_UINT32_MAX)
		estimated_pairs = PG_UINT32_MAX;
	pairhash = aftppair_create(CurrentMemoryContext,
							   (uint32) estimated_pairs, NULL);

	/*
	 * Process each query trace.  For each pair of same-type entries
	 * within the sliding window W, compute a time-decayed affinity
	 * contribution and add it to the symmetric pair key in the hash.
	 */
	for (int q = 0; q < traces->n_completed; q++)
	{
		AftpQueryTrace *trace = &traces->traces[q];
		int			n = trace->nentries;
		AftpTraceEntry *entries = trace->entries;

		for (int i = 0; i < n; i++)
		{
			AftpTraceEntry *ei = &entries[i];

			if (ei->type != target_type)
				continue;

			{
				int			max_j = Min(i + effective_window, n);
				uint32		ti = ei->tuple_id;

				for (int j = i + 1; j < max_j; j++)
				{
					AftpTraceEntry *ej = &entries[j];
					uint32		tj;
					uint32		id1;
					uint32		id2;
					uint64		pair_key;
					float		decay;
					AftpPairEntry *h_entry;
					bool		found;

					if (ej->type != target_type)
						continue;

					tj = ej->tuple_id;
					if (ti == tj)
						continue;

					/*
					 * Build symmetric key: always use (min << 32) | max
					 * so that (ti,tj) and (tj,ti) map to the same
					 * entry.
					 */
					if (ti < tj)
					{
						id1 = ti;
						id2 = tj;
					}
					else
					{
						id1 = tj;
						id2 = ti;
					}

					pair_key = ((uint64) id1 << 32) | (uint64) id2;

						/* Bounds check: id1, id2 are used as array indices below */
						if (id1 >= (uint32) n_tuples || id2 >= (uint32) n_tuples)
							elog(ERROR, "AFTP Phase 2: tuple_id out of range "
								 "(id1=%u, id2=%u, n_tuples=%d)",
								 id1, id2, n_tuples);

						decay = expf(-(float) (j - i) / effective_lambda);

					h_entry = aftppair_insert(pairhash, pair_key, &found);
					if (found)
						h_entry->affinity += decay;
					else
						h_entry->affinity = decay;
				}
			}
		}
	}

	hash_members = pairhash->members;

	/* ---- Step 3: Initialize bounded per-tuple heaps ---- */
	for (int i = 0; i < n_tuples; i++)
	{
		A->rows[i].tuple_id = (uint32) i;
		A->rows[i].max_entries = (uint32) top_affinity;
		A->rows[i].n_entries = 0;
		A->rows[i].entries = NULL;
	}

	/* ---- Step 4: Accumulate all weights and retain each row's top-K ---- */
	{
		aftppair_iterator iter;
		AftpPairEntry *e;

		aftppair_start_iterate(pairhash, &iter);
		while ((e = aftppair_iterate(pairhash, &iter)) != NULL)
		{
			uint32		id1 = (uint32) (e->pair_key >> 32);
			uint32		id2 = (uint32) (e->pair_key & 0xFFFFFFFF);
			float		aff = e->affinity;

			/* Accumulate point weight (sum of ALL affinities, not just top-K) */
			A->point_weights[id1] += aff;
			A->point_weights[id2] += aff;

			AftpAffinityHeapOffer(&A->rows[id1], heap_capacity, id2, aff);
			AftpAffinityHeapOffer(&A->rows[id2], heap_capacity, id1, aff);
		}
	}

	/* ---- Step 5: Sort retained heap entries into final row order ---- */
	for (int i = 0; i < n_tuples; i++)
	{
		AftpAffinityRow *row = &A->rows[i];

		if (row->n_entries > 1)
			qsort(row->entries, row->n_entries, sizeof(AftpAffinityEntry),
				  compare_affinity_desc);
	}

	/* ---- Step 6: Free the global accumulator ---- */
	aftppair_destroy(pairhash);

	elog(INFO, "AFTP Phase 2: affinity matrix built (n_tuples=%d, "
		 "K=%d, unique_pairs=%u)",
		 n_tuples, top_affinity, hash_members);

	return A;
}

/* ----------------------------------------------------------------
 * AftpFreeAffinityMatrix
 *
 * Free all dynamically allocated memory in the affinity matrix.
 * ---------------------------------------------------------------- */
void
AftpFreeAffinityMatrix(AftpAffinityMatrix *A)
{
	if (A == NULL)
		return;

	if (A->rows != NULL)
	{
		for (uint32 i = 0; i < A->n_tuples; i++)
		{
			if (A->rows[i].entries != NULL)
				pfree(A->rows[i].entries);
		}
		pfree(A->rows);
	}

	if (A->point_weights != NULL)
		pfree(A->point_weights);

	pfree(A);
}

/* ==================================================================
 * Phase 3: Greedy Seed-and-Grow Packing
 *
 * Implements Algorithm 3 from the design document (Section 3.4):
 *   1. Sort all tuples by point_weight descending
 *   2. Iteratively select seed tuples (deterministic or softmax-sampled)
 *   3. Grow each page by greedily adding the candidate with the highest
 *      total affinity to current page members
 *   4. Remaining zero-affinity tuples each get their own page
 *
 * Output:
 *   page_assignment[t] = page_id for each tuple t
 *   pages[] = array of AftpPage pointers
 *   n_pages = total page count
 * ================================================================== */

/* Temporary struct for sorting tuples by point_weight */
typedef struct
{
	uint32		tuple_id;
	float		weight;
}			AftpWeightedTuple;

/*
 * Remaining tuples in sorted-position order.  Nodes are only removed, so an
 * array-backed doubly linked list gives O(1) access to the first tuple, O(L)
 * traversal of the first L tuples, and O(1) removal by tuple ID or position.
 */
typedef struct
{
	uint32		n_positions;
	uint32		head;
	uint32		tail;
	uint32	   *next;
	uint32	   *prev;
	uint32	   *tuple_positions;
	int			count;
}			AftpUnplacedList;

#define AFTP_NO_POSITION PG_UINT32_MAX

/* Comparison function for sorting by weight descending */
static int
compare_weighted_desc(const void *a, const void *b)
{
	const AftpWeightedTuple *wa = (const AftpWeightedTuple *) a;
	const AftpWeightedTuple *wb = (const AftpWeightedTuple *) b;

	if (wa->weight > wb->weight)
		return -1;
	if (wa->weight < wb->weight)
		return 1;
	return 0;
}

/* Remove one live sorted position from the unplaced list. */
static void
AftpRemoveUnplaced(AftpUnplacedList *unplaced,
				   AftpWeightedTuple *sorted, uint32 position)
{
	uint32		prev;
	uint32		next;
	uint32		tuple_id;

	Assert(position < unplaced->n_positions);
	prev = unplaced->prev[position];
	next = unplaced->next[position];
	tuple_id = sorted[position].tuple_id;
	Assert(tuple_id < unplaced->n_positions);
	Assert(unplaced->tuple_positions[tuple_id] == position);
	Assert(unplaced->count > 0);

	if (prev == AFTP_NO_POSITION)
		unplaced->head = next;
	else
		unplaced->next[prev] = next;

	if (next == AFTP_NO_POSITION)
		unplaced->tail = prev;
	else
		unplaced->prev[next] = prev;

	unplaced->next[position] = AFTP_NO_POSITION;
	unplaced->prev[position] = AFTP_NO_POSITION;
	unplaced->tuple_positions[tuple_id] = AFTP_NO_POSITION;
	unplaced->count--;

	Assert((unplaced->count == 0) ==
		   (unplaced->head == AFTP_NO_POSITION));
	Assert((unplaced->count == 0) ==
		   (unplaced->tail == AFTP_NO_POSITION));
}

/* Grow the page-pointer array without overflowing int or Size. */
static void
AftpEnsurePageCapacity(AftpPage ***pages, int *capacity, int needed)
{
	Size		max_capacity = Min((Size) PG_INT32_MAX,
								   MaxAllocSize / sizeof(AftpPage *));
	Size		new_capacity;

	if (needed <= *capacity)
		return;
	if (needed < 0 || (Size) needed > max_capacity)
		elog(ERROR, "AFTP Phase 3: too many output pages");

	if ((Size) *capacity > max_capacity / 2)
		new_capacity = max_capacity;
	else
		new_capacity = (Size) *capacity * 2;

	if (new_capacity < (Size) needed)
		new_capacity = needed;

	*pages = repalloc(*pages, new_capacity * sizeof(AftpPage *));
	*capacity = (int) new_capacity;
}

/* ----------------------------------------------------------------
 * LookupAffinity
 *
 * Look up A(t_i, t_j) in the sparse affinity matrix.
 * Entries in each row are sorted by affinity descending, so we
 * perform a linear scan. K_affinity is typically 20, so O(K) per
 * lookup is acceptable in practice.
 *
 * Returns 0.0f if no affinity entry exists.
 * ---------------------------------------------------------------- */
static float
LookupAffinity(AftpAffinityMatrix *A, uint32 t_i, uint32 t_j)
{
	AftpAffinityRow *row = &A->rows[t_i];

	for (uint32 i = 0; i < row->n_entries; i++)
	{
		if (row->entries[i].tuple_id == t_j)
			return row->entries[i].affinity;
	}
	return 0.0f;
}

/* ----------------------------------------------------------------
 * softmax_sample (static helper)
 *
 * Given an array of (weight, tuple_id) pairs, softmax-sample one
 * where probability is proportional to exp(weight / temperature).
 *
 * Returns the index into the candidates array.
 * ---------------------------------------------------------------- */
static int
softmax_sample(AftpWeightedTuple *candidates, int n_cand, float temperature)
{
	float		total = 0.0f;
	float	   *probs;
	float		r;
	float		accum = 0.0f;

	if (n_cand <= 0)
		return 0;
	if (temperature <= 0.0f)
		temperature = 1.0f;

	probs = palloc(n_cand * sizeof(float));

	for (int i = 0; i < n_cand; i++)
	{
		probs[i] = expf(candidates[i].weight / temperature);
		total += probs[i];
	}

	if (total <= 0.0f)
	{
		/* All weights are zero -- pick uniformly */
		pfree(probs);
		return (int) (RandomDouble() * n_cand);
	}

	r = (float) RandomDouble() * total;
	for (int i = 0; i < n_cand; i++)
	{
		accum += probs[i];
		if (r <= accum)
		{
			pfree(probs);
			return i;
		}
	}

	pfree(probs);
	return n_cand - 1;
}

/* ----------------------------------------------------------------
 * AftpComputePageCapacity
 *
 * Compute the number of ElementTuples or NeighborTuples that fit in
 * one 8KB PostgreSQL page (BLCKSZ).
 *
 * Walks the graph->head linked list, computes the average tuple size
 * for the requested tuple type, and returns floor(BLCKSZ / avg_size).
 * ---------------------------------------------------------------- */
int
AftpComputePageCapacity(HnswBuildState *buildstate, bool is_element_tuple)
{
	char	   *base = buildstate->hnswarea;
	HnswElement element;
	double		total_size = 0.0;
	int			count = 0;
	int			capacity;

	element = HnswPtrAccess(base, buildstate->graph->head);

	while (element != NULL)
	{
		if (is_element_tuple)
		{
			total_size += HNSW_ELEMENT_TUPLE_SIZE(buildstate->dimensions *
												  sizeof(float));
		}
		else
		{
			total_size += HNSW_NEIGHBOR_TUPLE_SIZE(element->level,
												   buildstate->m);
		}
		count++;
		element = HnswPtrAccess(base, element->next);
	}

	if (count == 0)
	{
		/* Fallback: use a single-element estimate */
		if (is_element_tuple)
			total_size = HNSW_ELEMENT_TUPLE_SIZE(buildstate->dimensions *
												 sizeof(float));
		else
			total_size = HNSW_NEIGHBOR_TUPLE_SIZE(0, buildstate->m);
		count = 1;
	}

	capacity = (int) (BLCKSZ / (total_size / count));

	if (capacity < 1)
		capacity = 1;

	return capacity;
}

/* ----------------------------------------------------------------
 * AftpGreedyPacking
 *
 * Main Phase 3 entry point.  Sorts all tuples by point_weight and
 * greedily packs them into pages using seed-and-grow.
 *
 * Algorithm:
 *   1. Sort all tuples by point_weight descending
 *      and link their sorted positions in an unplaced list.
 *   2. While unplaced tuples remain:
 *      a. If the max remaining weight <= 0, place each remaining
 *         tuple in its own singleton page and stop.
 *      b. Select seed: either deterministic (first unplaced = max
 *         weight) or softmax-sampled from top seed_softmax candidates.
 *      c. Create a new page with the seed.
 *      d. Grow the page: enumerate the top candidates from the
 *         unplaced list (limit = 10*K or sqrt(unplaced)) and greedily
 *         add the one with highest sum-of-affinity to page members.
 *      e. Finalize the page.
 *   3. Output page_assignment[], pages[], n_pages.
 *
 * Parameters:
 *   A               - Affinity matrix (from Phase 2)
 *   K               - Page capacity (max tuples per page)
 *   seed_softmax    - Softmax window for seed selection
 *                     0 = always pick max-weight unplaced tuple
 *   out_page_assignment - Output: page_assignment[t] = page_id
 *   out_pages       - Output: array of AftpPage pointers
 *   out_n_pages     - Output: total number of pages
 * ---------------------------------------------------------------- */
void
AftpGreedyPacking(AftpAffinityMatrix *A, int K, int seed_softmax,
				  int **out_page_assignment, AftpPage ***out_pages,
				  int *out_n_pages)
{
	uint32		N = A->n_tuples;
	AftpWeightedTuple *sorted;
	AftpUnplacedList unplaced;
	int			n_pages = 0;
	AftpPage  **pages;
	int		   *page_assignment;
	int			pages_capacity = 64;

	if (N == 0)
	{
		*out_page_assignment = NULL;
		*out_pages = NULL;
		*out_n_pages = 0;
		return;
	}
	if (K <= 0)
		elog(ERROR, "AFTP Phase 3: page capacity must be positive (K=%d)", K);
	if ((Size) N > MaxAllocSize / sizeof(AftpWeightedTuple) ||
		(Size) N > MaxAllocSize / sizeof(uint32) ||
		(Size) N > MaxAllocSize / sizeof(int) ||
		(Size) K > MaxAllocSize / sizeof(uint32))
		elog(ERROR, "AFTP Phase 3: packing allocation is too large "
			 "(n_tuples=%u, K=%d)", N, K);

	/* ---- Step 1: Sort all tuples by point_weight descending ---- */
	sorted = palloc(N * sizeof(AftpWeightedTuple));
	for (uint32 i = 0; i < N; i++)
	{
		sorted[i].tuple_id = i;
		sorted[i].weight = A->point_weights[i];
	}
	qsort(sorted, N, sizeof(AftpWeightedTuple), compare_weighted_desc);

	/* ---- Step 2: Link all sorted positions as initially unplaced ---- */
	unplaced.n_positions = N;
	unplaced.head = 0;
	unplaced.tail = N - 1;
	unplaced.next = palloc(N * sizeof(uint32));
	unplaced.prev = palloc(N * sizeof(uint32));
	unplaced.tuple_positions = palloc(N * sizeof(uint32));
	unplaced.count = (int) N;

	for (uint32 i = 0; i < N; i++)
	{
		unplaced.prev[i] = (i == 0) ? AFTP_NO_POSITION : i - 1;
		unplaced.next[i] = (i + 1 == N) ? AFTP_NO_POSITION : i + 1;
		unplaced.tuple_positions[sorted[i].tuple_id] = i;
	}

	/* ---- Allocate output arrays ---- */
	page_assignment = palloc(N * sizeof(int));
	for (uint32 i = 0; i < N; i++)
		page_assignment[i] = -1;

	pages = palloc(pages_capacity * sizeof(AftpPage *));

	/* ---- Main loop: seed-and-grow ---- */
	while (unplaced.count > 0)
	{
		AftpPage   *page;
		int			scan_limit;
		uint32		best_candidate_position;
		float		best_score;

		/* --------------------------------------------------------
		 * Check if all remaining tuples have zero/negative affinity.
		 * If so, place each in its own singleton page and exit the
		 * main loop — Section 3.4 Step 3.
		 * -------------------------------------------------------- */
		{
			uint32		first_unplaced = unplaced.head;

			if (first_unplaced != AFTP_NO_POSITION &&
				A->point_weights[sorted[first_unplaced].tuple_id] <= 0.0f)
			{
				for (uint32 i = 0; i < N; i++)
				{
					uint32		position = unplaced.tuple_positions[i];

					if (position != AFTP_NO_POSITION)
					{
						AftpPage *zpage;

						zpage = palloc(sizeof(AftpPage));
						zpage->page_id = n_pages;
						zpage->n_tuples = 1;
						zpage->capacity = K;
						zpage->tuple_ids = palloc(sizeof(uint32));
						zpage->tuple_ids[0] = i;

						AftpEnsurePageCapacity(&pages, &pages_capacity,
										   n_pages + 1);
						pages[n_pages] = zpage;
						page_assignment[i] = n_pages;
						n_pages++;
						AftpRemoveUnplaced(&unplaced, sorted, position);
					}
				}
				Assert(unplaced.count == 0);
				break;
			}
		}

		/* === Seed Selection (§3.4 Step 2a) === */
		{
			uint32		seed_id;

			if (seed_softmax > 0 && unplaced.count > 1)
			{
				/* Softmax-sample from top seed_softmax unplaced */
				int			n_cand = Min(seed_softmax, unplaced.count);
				AftpWeightedTuple *candidates;
				uint32		position = unplaced.head;
				int			selected;

				candidates = palloc(n_cand * sizeof(AftpWeightedTuple));

				for (int ci = 0; ci < n_cand; ci++)
				{
					Assert(position != AFTP_NO_POSITION);
					candidates[ci] = sorted[position];
					position = unplaced.next[position];
				}

				selected = softmax_sample(candidates, n_cand, 1.0f);
				seed_id = candidates[selected].tuple_id;
				pfree(candidates);
			}
			else
			{
				/* Deterministic: pick first unplaced (max weight) */
				if (unplaced.head == AFTP_NO_POSITION)
					break;	/* safety: all placed */

				seed_id = sorted[unplaced.head].tuple_id;
			}

			/* Create new page with seed (§3.4 Step 2b) */
			page = palloc(sizeof(AftpPage));
			page->page_id = n_pages;
			page->n_tuples = 1;
			page->capacity = K;
			page->tuple_ids = palloc(K * sizeof(uint32));
			page->tuple_ids[0] = seed_id;
			AftpRemoveUnplaced(&unplaced, sorted,
							   unplaced.tuple_positions[seed_id]);
		}

		/* === Grow Page (§3.4 Step 2c) === */
		while (page->n_tuples < K && unplaced.count > 0)
		{
			int			base_limit = Max(K, (int) sqrt((double) unplaced.count));

			scan_limit = (int) Min((int64) 10 * K, (int64) base_limit);

			best_score = 0.0f;
			best_candidate_position = AFTP_NO_POSITION;

			/* Enumerate the top scan_limit live sorted positions. */
			{
				int			scanned = 0;
				uint32		position = unplaced.head;

				while (position != AFTP_NO_POSITION && scanned < scan_limit)
				{
					uint32		candidate_position = position;
					uint32		cand_id = sorted[candidate_position].tuple_id;

					position = unplaced.next[candidate_position];
					scanned++;

					if (best_score > 0.0f &&
						A->point_weights[cand_id] == 0.0f)
					{
						/*
						 * Heuristic: if we already found a positive
						 * affinity candidate, skip zero-weight tuples
						 * as they cannot contribute positive affinity.
						 */
						continue;
					}

					{
						float		score = 0.0f;

						for (int m = 0; m < page->n_tuples; m++)
						{
							score += LookupAffinity(A, cand_id,
													page->tuple_ids[m]);
						}

						if (score > best_score)
						{
							best_score = score;
							best_candidate_position = candidate_position;
						}
					}
				}
			}

			/* If no candidate has positive affinity, stop growing */
			if (best_candidate_position == AFTP_NO_POSITION)
				break;

			/* Add best candidate to page (§3.4 Step 2c cont.) */
			{
				uint32		cand_id = sorted[best_candidate_position].tuple_id;

				page->tuple_ids[page->n_tuples++] = cand_id;
				AftpRemoveUnplaced(&unplaced, sorted,
								   best_candidate_position);
			}
		}

		/* === Finalize Page (§3.4 Step 2d) === */
		AftpEnsurePageCapacity(&pages, &pages_capacity, n_pages + 1);
		pages[n_pages] = page;

		for (int m = 0; m < page->n_tuples; m++)
			page_assignment[page->tuple_ids[m]] = n_pages;

		n_pages++;
	}

	*out_page_assignment = page_assignment;
	*out_pages = pages;
	*out_n_pages = n_pages;

	pfree(sorted);
	pfree(unplaced.next);
	pfree(unplaced.prev);
	pfree(unplaced.tuple_positions);
}

/* ----------------------------------------------------------------
 * AftpComputeLocalityScore
 *
 * Compute LS(P) = sum over all pages of sum of affinities within page.
 *
 * Iterates all tuples and for each of its affinity entries, checks if
 * the partner is in the same page.  Divides by 2 because each pair is
 * counted twice (once from each direction in the symmetric matrix).
 *
 * Returns the locality score (float).
 * ---------------------------------------------------------------- */
float
AftpComputeLocalityScore(AftpAffinityMatrix *A, int *P, int n_tuples)
{
	float		total = 0.0f;

	for (int i = 0; i < n_tuples; i++)
	{
		AftpAffinityRow *row = &A->rows[i];
		int			pi = P[i];

		if (pi < 0)
			continue;		/* Unassigned tuple -- skip */

		for (uint32 k = 0; k < row->n_entries; k++)
		{
			uint32		j = row->entries[k].tuple_id;

			if ((int) j < n_tuples && P[j] == pi)
				total += row->entries[k].affinity;
		}
	}

	/* Each pair counted twice (once from each direction) */
	return total / 2.0f;
}

/* ==================================================================
 * Phase 4: Iterative BNF-style Refinement (§3.5, Algorithm 4)
 *
 * After greedy packing, each tuple is in a page with some set of
 * partners.  Phase 4 iteratively examines each tuple and considers
 * moving it to a better page — i.e., a page that contains more of
 * its affinity partners — using BNF-style (Best-Neighbor-First)
 * single-tuple moves.
 *
 * Algorithm (simplified):
 *   for iter = 1 to max_iterations:
 *     Build inverted index (page -> external tuples with affinity)
 *     for each tuple t:
 *       current_score = sum affinities with current page members
 *       best_page = current_page, best_gain = 0
 *       for each candidate page p (where t has affinity partners
 *            AND has space):
 *         new_score = sum affinities with page p members
 *         gain = new_score - current_score
 *         if gain > best_gain: best_page = p, best_gain = gain
 *       if best_page != current_page: move t
 *     if no moves: converged
 *
 * Simplifications (per task spec):
 *   1. Only consider pages where t's affinity partners reside
 *   2. Sequential processing (no shuffle)
 *   3. Inverted index rebuilt each iteration
 *   4. Early termination when n_moves == 0
 * ================================================================== */

/* ----------------------------------------------------------------
 * Page helper: remove a tuple from a page's tuple_ids array.
 * Shifts subsequent elements down to close the gap.
 * ---------------------------------------------------------------- */
static void
page_remove_tuple(AftpPage *page, uint32 tuple_id)
{
	int			idx = -1;

	for (int i = 0; i < page->n_tuples; i++)
	{
		if (page->tuple_ids[i] == tuple_id)
		{
			idx = i;
			break;
		}
	}

	if (idx < 0)
		return;					/* not found */

	for (int i = idx; i < page->n_tuples - 1; i++)
		page->tuple_ids[i] = page->tuple_ids[i + 1];

	page->n_tuples--;
}

/* ----------------------------------------------------------------
 * Page helper: add a tuple to a page's tuple_ids array.
 * Requires page->n_tuples < page->capacity (caller must check).
 * ---------------------------------------------------------------- */
static void
page_add_tuple(AftpPage *page, uint32 tuple_id)
{
	if (page->n_tuples >= page->capacity)
		elog(ERROR, "AFTP Phase 4: page %d is full (n_tuples=%d, capacity=%d)",
			 page->page_id, page->n_tuples, page->capacity);

	page->tuple_ids[page->n_tuples++] = tuple_id;
}

/* ----------------------------------------------------------------
 * AftpBuildInvertedIndex
 *
 * Build a page -> external-tuple reverse index from the affinity
 * matrix and current page assignment.
 *
 * page_partners[p]  = array of tuple IDs t such that t has non-zero
 *                      affinity with at least one tuple in page p.
 * page_npartners[p] = number of such external tuples.
 *
 * Implementation:
 *   Step 1: Count partners per page (uint16 per page — overflow
 *           impossible because K * K_affinity ~ 200*20 = 4000).
 *   Step 2: Allocate per-page arrays.
 *   Step 3: Fill with dedup (each tuple appears at most once per
 *           page).  Uses a generation counter instead of memset
 *           for O(1) reset per tuple.
 * ---------------------------------------------------------------- */
static AftpInvertedIndex *
AftpBuildInvertedIndex(AftpAffinityMatrix *A, int *P, int n_tuples,
					   int n_pages, int top_affinity)
{
	uint16	   *counts;
	uint32	  **page_partners;
	uint16	   *page_npartners;
	AftpInvertedIndex *index;

	(void) top_affinity;		/* unused in this implementation */

	if (n_pages <= 0)
	{
		index = palloc(sizeof(AftpInvertedIndex));
		index->page_partners = NULL;
		index->page_npartners = NULL;
		index->n_pages = 0;
		return index;
	}

	/* Step 1: Count partners per page */
	counts = palloc0(n_pages * sizeof(uint16));
	for (uint32 t = 0; t < (uint32) n_tuples; t++)
	{
		AftpAffinityRow *row = &A->rows[t];

		if (row->n_entries == 0)
			continue;

		for (uint32 k = 0; k < row->n_entries; k++)
		{
			uint32		partner = row->entries[k].tuple_id;
			int			pp = P[partner];

			if (pp < 0 || pp >= n_pages)
				continue;

			counts[pp]++;
		}
	}

	/* Step 2: Allocate per-page arrays */
	page_partners = palloc(n_pages * sizeof(uint32 *));
	page_npartners = palloc(n_pages * sizeof(uint16));

	for (int p = 0; p < n_pages; p++)
	{
		page_npartners[p] = counts[p];
		if (counts[p] > 0)
			page_partners[p] = palloc(counts[p] * sizeof(uint32));
		else
			page_partners[p] = NULL;
		counts[p] = 0;			/* reset for fill pass */
	}

	/* Step 3: Fill with dedup (generation counter) */
	{
		uint32	   *seen_gen = palloc0(n_pages * sizeof(uint32));
		uint32		gen = 1;

		for (uint32 t = 0; t < (uint32) n_tuples; t++)
		{
			AftpAffinityRow *row = &A->rows[t];

			if (row->n_entries == 0)
				continue;

			for (uint32 k = 0; k < row->n_entries; k++)
			{
				uint32		partner = row->entries[k].tuple_id;
				int			pp = P[partner];

				if (pp < 0 || pp >= n_pages)
					continue;

				if (seen_gen[pp] != gen)
				{
					page_partners[pp][counts[pp]++] = t;
					seen_gen[pp] = gen;
				}
			}
			gen++;
		}

		pfree(seen_gen);
	}

	pfree(counts);

	/* Build result struct */
	index = palloc(sizeof(AftpInvertedIndex));
	index->page_partners = page_partners;
	index->page_npartners = page_npartners;
	index->n_pages = n_pages;

	return index;
}

/* ----------------------------------------------------------------
 * AftpFreeInvertedIndex
 *
 * Free all memory associated with an inverted index.
 * ---------------------------------------------------------------- */
static void
AftpFreeInvertedIndex(AftpInvertedIndex *index)
{
	if (index == NULL)
		return;

	for (int p = 0; p < index->n_pages; p++)
	{
		if (index->page_partners[p] != NULL)
			pfree(index->page_partners[p]);
	}

	if (index->page_partners != NULL)
		pfree(index->page_partners);
	if (index->page_npartners != NULL)
		pfree(index->page_npartners);

	pfree(index);
}

/* ----------------------------------------------------------------
 * AftpIterativeRefinement
 *
 * Main Phase 4 entry point.  Iteratively examines each tuple and
 * considers moving it to a page with stronger total affinity.
 *
 * Simplifications per the task specification:
 *   - Candidate pages: only pages where t has at least one affinity
 *     partner (determined by scanning t's affinity row), plus pages
 *     with empty slots (not full).  No pair-exchange between full
 *     pages.
 *   - Sequential processing: tuples are processed in order 0..N-1.
 *   - Inverted index is rebuilt after each full iteration.
 *   - Convergence: stops when n_moves == 0.
 * ---------------------------------------------------------------- */
void
AftpIterativeRefinement(AftpAffinityMatrix *A, int *P, AftpPage **pages,
						int n_pages, int n_tuples, int max_iterations, int K)
{
	if (n_tuples <= 0 || n_pages <= 0 || max_iterations <= 0)
		return;

	elog(INFO, "AFTP Phase 4: starting iterative refinement "
		 "(max_iter=%d, n_tuples=%d, n_pages=%d, K=%d)",
		 max_iterations, n_tuples, n_pages, K);

	for (int iter = 0; iter < max_iterations; iter++)
	{
		int					 n_moves = 0;
		AftpInvertedIndex	*inverted;
		uint32				*page_seen_gen;
		uint32				 gen;

		/* Build inverted index for this iteration */
		inverted = AftpBuildInvertedIndex(A, P, n_tuples, n_pages,
										  (int) A->max_entries);

		/*
		 * Generation counter for O(1) per-tuple dedup of candidate
		 * pages.  Allocated once per iteration, avoids memset of a
		 * potentially large array.
		 */
		page_seen_gen = palloc0(n_pages * sizeof(uint32));
		gen = 1;

		/* Process each tuple sequentially */
		for (uint32 t = 0; t < (uint32) n_tuples; t++)
		{
			int			current_page = P[t];
			float		current_score = 0.0f;
			AftpPage   *cur_page;
			int			best_page;
			float		best_gain;
			AftpAffinityRow *row;

			if (current_page < 0 || current_page >= n_pages)
				continue;

			cur_page = pages[current_page];

			/* Sanity: tuple must be in its recorded page */
			{
				bool		found = false;

				for (int m = 0; m < cur_page->n_tuples; m++)
				{
					if (cur_page->tuple_ids[m] == t)
					{
						found = true;
						break;
					}
				}
				if (!found)
					continue;
			}

			/* Compute current score */
			for (int m = 0; m < cur_page->n_tuples; m++)
			{
				uint32		u = cur_page->tuple_ids[m];

				if (u != t)
					current_score += LookupAffinity(A, t, u);
			}

			best_page = current_page;
			best_gain = 0.0f;

			/* Scan t's affinity entries to find candidate pages */
			row = &A->rows[t];
			for (uint32 k = 0; k < row->n_entries; k++)
			{
				uint32		partner = row->entries[k].tuple_id;
				int			pp = P[partner];

				if (pp < 0 || pp == current_page || pp >= n_pages)
					continue;

				/* Dedup: already considered this page for this tuple */
				if (page_seen_gen[pp] == gen)
					continue;
				page_seen_gen[pp] = gen;

				/* Skip full pages */
				if (pages[pp]->n_tuples >= K)
					continue;

				/* Compute new score */
				{
					float		new_score = 0.0f;
					float		gain;
					AftpPage   *cand_page = pages[pp];

					for (int m = 0; m < cand_page->n_tuples; m++)
						new_score += LookupAffinity(A, t,
													cand_page->tuple_ids[m]);

					gain = new_score - current_score;

					if (gain > best_gain)
					{
						best_gain = gain;
						best_page = pp;
					}
				}
			}

			gen++;

			/* Move if beneficial */
			if (best_page != current_page)
			{
				page_remove_tuple(pages[current_page], t);
				page_add_tuple(pages[best_page], t);
				P[t] = best_page;
				n_moves++;
			}
		}

		pfree(page_seen_gen);
		AftpFreeInvertedIndex(inverted);

		elog(INFO, "AFTP Phase 4: iteration %d/%d, n_moves=%d",
			 iter + 1, max_iterations, n_moves);

		if (n_moves == 0)
		{
			elog(INFO, "AFTP Phase 4: converged after %d iterations",
				 iter + 1);
			break;
		}
	}
}

/* ----------------------------------------------------------------
 * AftpPopulateConfig
 *
 * Populate an AftpConfig struct from GUC variables.
 * Called by HnswBuildState initialization to snapshot GUC values
 * at build start time.
 * ---------------------------------------------------------------- */
void
AftpPopulateConfig(AftpConfig *cfg)
{
	cfg->enabled = aftp_enable;
	cfg->sample_queries = aftp_sample_queries;
	cfg->ef_search = aftp_ef_search;
	cfg->window = aftp_window;
	cfg->top_affinity = aftp_top_affinity;
	cfg->refine_iterations = aftp_refine_iterations;
	cfg->decay_lambda = aftp_decay_lambda;
	cfg->seed_softmax = aftp_seed_softmax;
	cfg->max_memory_mb = aftp_max_memory;
}

/* ==================================================================
 * AFTP_Main Pipeline
 *
 * Orchestrates all four phases of the AFTP algorithm:
 *   Phase 1: Trace collection
 *   Phase 2: Affinity matrix construction (ElementTuple + NeighborTuple)
 *   Phase 3: Greedy seed-and-grow packing
 *   Phase 4: Iterative BNF-style refinement (optional)
 *
 * Stores results in buildstate:
 *   aftp_etuple_page  -- P_E[t] = page_id for each ElementTuple
 *   aftp_ntuple_page  -- P_N[t] = page_id for each NeighborTuple
 *   aftp_etuple_order -- ElementTuple write order (page-major)
 *   aftp_ntuple_order -- NeighborTuple write order (page-major)
 *   aftp_n_epages     -- Total ElementTuple page count
 *   aftp_n_npages     -- Total NeighborTuple page count
 *
 * Free policy:
 *   - Trace state and affinity matrices are freed inside this function.
 *   - Packing results (pages_E, pages_N, P_E, P_N, write orders)
 *     are kept for CreateGraphPagesPacked.
 * ================================================================== */
void
AftpMain(HnswBuildState *buildstate, AftpConfig *config)
{
	AftpTraceState *traces;
	AftpAffinityMatrix *A_E;
	AftpAffinityMatrix *A_N;
	int			K_E;
	int			K_N;
	int		   *P_E;
	int		   *P_N;
	AftpPage  **pages_E;
	AftpPage  **pages_N;
	int			n_epages;
	int			n_npages;
	float		LS_E;
	float		LS_N;
	instr_time	pipeline_start;
	instr_time	phase_start;
	double		phase1_ms;
	double		phase2_element_ms;
	double		phase2_neighbor_ms;
	double		phase3_capacity_ms;
	double		phase3_element_ms;
	double		phase3_neighbor_ms;
	double		phase4_element_ms = 0.0;
	double		phase4_neighbor_ms = 0.0;

	elog(INFO, "AFTP: starting AFTP_Main pipeline");
	INSTR_TIME_SET_CURRENT(pipeline_start);

	/* --------------------------------------------------------
	 * Phase 1: Trace collection
	 * -------------------------------------------------------- */
	INSTR_TIME_SET_CURRENT(phase_start);
	traces = AftpCollectTraces(buildstate, config);
	phase1_ms = AftpElapsedMilliseconds(phase_start);
	elog(INFO, "AFTP timing: Phase 1 trace collection: %.3f ms",
		 phase1_ms);
	if (traces == NULL || traces->n_completed == 0)
	{
		elog(WARNING, "AFTP: trace collection produced no traces, aborting");
		elog(INFO, "AFTP timing: pipeline aborted after %.3f ms",
			 AftpElapsedMilliseconds(pipeline_start));
		return;
	}

	/* --------------------------------------------------------
	 * Phase 2: Affinity matrices
	 * -------------------------------------------------------- */
	INSTR_TIME_SET_CURRENT(phase_start);
	A_E = AftpBuildAffinityMatrix(traces, traces->n_elements,
								  config->top_affinity, config->window,
								  config->decay_lambda, AFTP_TUPLE_ELEMENT);
	phase2_element_ms = AftpElapsedMilliseconds(phase_start);
	elog(INFO, "AFTP timing: Phase 2 ElementTuple affinity: %.3f ms",
		 phase2_element_ms);
	if (A_E == NULL)
	{
		elog(WARNING, "AFTP: failed to build ElementTuple affinity matrix");
		AftpFreeTraceState(traces);
		elog(INFO, "AFTP timing: pipeline aborted after %.3f ms",
			 AftpElapsedMilliseconds(pipeline_start));
		return;
	}

	INSTR_TIME_SET_CURRENT(phase_start);
	A_N = AftpBuildAffinityMatrix(traces, traces->n_neighbors,
								  config->top_affinity, config->window,
								  config->decay_lambda, AFTP_TUPLE_NEIGHBOR);
	phase2_neighbor_ms = AftpElapsedMilliseconds(phase_start);
	elog(INFO, "AFTP timing: Phase 2 NeighborTuple affinity: %.3f ms "
		 "(total Phase 2: %.3f ms)",
		 phase2_neighbor_ms, phase2_element_ms + phase2_neighbor_ms);
	if (A_N == NULL)
	{
		elog(WARNING, "AFTP: failed to build NeighborTuple affinity matrix");
		AftpFreeAffinityMatrix(A_E);
		AftpFreeTraceState(traces);
		elog(INFO, "AFTP timing: pipeline aborted after %.3f ms",
			 AftpElapsedMilliseconds(pipeline_start));
		return;
	}

	/* --------------------------------------------------------
	 * Phase 3: Compute page capacities
	 * -------------------------------------------------------- */
	INSTR_TIME_SET_CURRENT(phase_start);
	K_E = AftpComputePageCapacity(buildstate, true);
	K_N = AftpComputePageCapacity(buildstate, false);
	phase3_capacity_ms = AftpElapsedMilliseconds(phase_start);
	elog(INFO, "AFTP timing: Phase 3 page capacity estimation: %.3f ms",
		 phase3_capacity_ms);

	elog(INFO, "AFTP: page capacities: K_E=%d, K_N=%d", K_E, K_N);

	/* --------------------------------------------------------
	 * Phase 3: Greedy seed-and-grow packing
	 * -------------------------------------------------------- */
	INSTR_TIME_SET_CURRENT(phase_start);
	AftpGreedyPacking(A_E, K_E, config->seed_softmax,
					  &P_E, &pages_E, &n_epages);
	phase3_element_ms = AftpElapsedMilliseconds(phase_start);
	elog(INFO, "AFTP timing: Phase 3 ElementTuple packing: %.3f ms",
		 phase3_element_ms);

	INSTR_TIME_SET_CURRENT(phase_start);
	AftpGreedyPacking(A_N, K_N, config->seed_softmax,
					  &P_N, &pages_N, &n_npages);
	phase3_neighbor_ms = AftpElapsedMilliseconds(phase_start);
	elog(INFO, "AFTP timing: Phase 3 NeighborTuple packing: %.3f ms "
		 "(total Phase 3: %.3f ms)",
		 phase3_neighbor_ms,
		 phase3_capacity_ms + phase3_element_ms + phase3_neighbor_ms);

	elog(INFO, "AFTP Phase 3: E-pages=%d, N-pages=%d", n_epages, n_npages);

	/* --------------------------------------------------------
	 * Phase 4: Iterative refinement (if enabled)
	 * -------------------------------------------------------- */
	if (config->refine_iterations > 0)
	{
		elog(INFO, "AFTP Phase 4: refining ElementTuple layout "
			 "(%d iterations)", config->refine_iterations);
		INSTR_TIME_SET_CURRENT(phase_start);
		AftpIterativeRefinement(A_E, P_E, pages_E, n_epages,
								traces->n_elements,
								config->refine_iterations, K_E);
		phase4_element_ms = AftpElapsedMilliseconds(phase_start);
		elog(INFO, "AFTP timing: Phase 4 ElementTuple refinement: %.3f ms",
			 phase4_element_ms);

		elog(INFO, "AFTP Phase 4: refining NeighborTuple layout "
			 "(%d iterations)", config->refine_iterations);
		INSTR_TIME_SET_CURRENT(phase_start);
		AftpIterativeRefinement(A_N, P_N, pages_N, n_npages,
								traces->n_neighbors,
								config->refine_iterations, K_N);
		phase4_neighbor_ms = AftpElapsedMilliseconds(phase_start);
		elog(INFO, "AFTP timing: Phase 4 NeighborTuple refinement: %.3f ms "
			 "(total Phase 4: %.3f ms)",
			 phase4_neighbor_ms, phase4_element_ms + phase4_neighbor_ms);
	}
	else
		elog(INFO, "AFTP timing: Phase 4 refinement skipped");

	/* --------------------------------------------------------
	 * Compute locality scores and log
	 * -------------------------------------------------------- */
	LS_E = AftpComputeLocalityScore(A_E, P_E, traces->n_elements);
	LS_N = AftpComputeLocalityScore(A_N, P_N, traces->n_neighbors);

	elog(INFO, "AFTP: ElementTuple LS=%.2f, NeighborTuple LS=%.2f, "
		 "E-pages=%d, N-pages=%d", (double) LS_E, (double) LS_N,
		 n_epages, n_npages);

	/* --------------------------------------------------------
	 * Store results in buildstate
	 * -------------------------------------------------------- */
	buildstate->aftp_etuple_page = P_E;
	buildstate->aftp_ntuple_page = P_N;
	buildstate->aftp_n_epages = n_epages;
	buildstate->aftp_n_npages = n_npages;

	/* --------------------------------------------------------
	 * Generate write orders (page-major: page_id then tuple_id)
	 * -------------------------------------------------------- */
	buildstate->aftp_etuple_order = palloc(traces->n_elements * sizeof(int));
	buildstate->aftp_ntuple_order = palloc(traces->n_neighbors * sizeof(int));

	{
		int			idx = 0;

		for (int p = 0; p < n_epages; p++)
			for (int t = 0; t < pages_E[p]->n_tuples; t++)
				buildstate->aftp_etuple_order[idx++] = pages_E[p]->tuple_ids[t];
	}

	{
		int			idx = 0;

		for (int p = 0; p < n_npages; p++)
			for (int t = 0; t < pages_N[p]->n_tuples; t++)
				buildstate->aftp_ntuple_order[idx++] = pages_N[p]->tuple_ids[t];
	}

	/* --------------------------------------------------------
	 * Free trace data and affinity matrices (keep packing results
	 * for CreateGraphPagesPacked)
	 * -------------------------------------------------------- */
	AftpFreeAffinityMatrix(A_E);
	AftpFreeAffinityMatrix(A_N);
	AftpFreeTraceState(traces);

	elog(INFO, "AFTP timing summary: Phase 1=%.3f ms, Phase 2=%.3f ms, "
		 "Phase 3=%.3f ms, Phase 4=%.3f ms, pipeline total=%.3f ms",
		 phase1_ms,
		 phase2_element_ms + phase2_neighbor_ms,
		 phase3_capacity_ms + phase3_element_ms + phase3_neighbor_ms,
		 phase4_element_ms + phase4_neighbor_ms,
		 AftpElapsedMilliseconds(pipeline_start));
	elog(INFO, "AFTP: AFTP_Main pipeline completed successfully");
}
