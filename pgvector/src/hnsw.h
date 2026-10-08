#ifndef HNSW_H
#define HNSW_H

#include "postgres.h"

#include "access/genam.h"
#include "access/parallel.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "port.h"				/* for random() */
#include "utils/relptr.h"
#include "utils/sampling.h"
#include "vector.h"
#include "pgstat.h"

#define HNSW_MAX_DIM 2000
#define HNSW_MAX_NNZ 1000

/* Support functions */
#define HNSW_DISTANCE_PROC 1
#define HNSW_NORM_PROC 2
#define HNSW_TYPE_INFO_PROC 3

#define HNSW_VERSION	1
#define HNSW_MAGIC_NUMBER 0xA953A953
#define HNSW_PAGE_ID	0xFF90

/* Preserved page numbers */
#define HNSW_METAPAGE_BLKNO	0
#define HNSW_HEAD_BLKNO		1	/* first element page */

/* Must correspond to page numbers since page lock is used */
#define HNSW_UPDATE_LOCK 	0
#define HNSW_SCAN_LOCK		1

/* HNSW parameters */
#define HNSW_DEFAULT_M	16
#define HNSW_MIN_M	2
#define HNSW_MAX_M		100
#define HNSW_DEFAULT_EF_CONSTRUCTION	64
#define HNSW_MIN_EF_CONSTRUCTION	4
#define HNSW_MAX_EF_CONSTRUCTION		1000
#define HNSW_DEFAULT_EF_SEARCH	40
#define HNSW_MIN_EF_SEARCH		1
#define HNSW_MAX_EF_SEARCH		1000

/* Tuple types */
#define HNSW_ELEMENT_TUPLE_TYPE  1
#define HNSW_NEIGHBOR_TUPLE_TYPE 2

/* Make graph robust against non-HOT updates */
#define HNSW_HEAPTIDS 10

#define HNSW_UPDATE_ENTRY_GREATER 1
#define HNSW_UPDATE_ENTRY_ALWAYS 2

/* Build phases */
/* PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE is 1 */
#define PROGRESS_HNSW_PHASE_LOAD		2

#define HNSW_MAX_SIZE (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - sizeof(ItemIdData))
#define HNSW_TUPLE_ALLOC_SIZE BLCKSZ

#define HNSW_ELEMENT_TUPLE_SIZE(size)	MAXALIGN(offsetof(HnswElementTupleData, data) + (size))
#define HNSW_NEIGHBOR_TUPLE_SIZE(level, m)	MAXALIGN(offsetof(HnswNeighborTupleData, indextids) + ((level) + 2) * (m) * sizeof(ItemPointerData))

#define HNSW_NEIGHBOR_ARRAY_SIZE(lm)	(offsetof(HnswNeighborArray, items) + sizeof(HnswCandidate) * (lm))

#define HnswPageGetOpaque(page)	((HnswPageOpaque) PageGetSpecialPointer(page))
#define HnswPageGetMeta(page)	((HnswMetaPageData *) PageGetContents(page))

#if PG_VERSION_NUM >= 150000
#define RandomDouble() pg_prng_double(&pg_global_prng_state)
#define SeedRandom(seed) pg_prng_seed(&pg_global_prng_state, seed)
#else
#define RandomDouble() (((double) random()) / MAX_RANDOM_VALUE)
#define SeedRandom(seed) srandom(seed)
#endif

#define HnswIsElementTuple(tup) ((tup)->type == HNSW_ELEMENT_TUPLE_TYPE)
#define HnswIsNeighborTuple(tup) ((tup)->type == HNSW_NEIGHBOR_TUPLE_TYPE)

/* 2 * M connections for ground layer */
#define HnswGetLayerM(m, layer) (layer == 0 ? (m) * 2 : (m))

/* Optimal ML from paper */
#define HnswGetMl(m) (1 / log(m))

/* Ensure fits on page and in uint8 */
#define HnswGetMaxLevel(m) Min(((BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - offsetof(HnswNeighborTupleData, indextids) - sizeof(ItemIdData)) / (sizeof(ItemPointerData)) / (m)) - 2, 255)

#define HnswGetSearchCandidate(membername, ptr) pairingheap_container(HnswSearchCandidate, membername, ptr)
#define HnswGetSearchCandidateConst(membername, ptr) pairingheap_const_container(HnswSearchCandidate, membername, ptr)

#define HnswGetValue(base, element) PointerGetDatum(HnswPtrAccess(base, (element)->value))

#if PG_VERSION_NUM < 140005
#define relptr_offset(rp) ((rp).relptr_off - 1)
#endif

/* Pointer macros */
#define HnswPtrAccess(base, hp) ((base) == NULL ? (hp).ptr : relptr_access(base, (hp).relptr))
#define HnswPtrStore(base, hp, value) ((base) == NULL ? (void) ((hp).ptr = (value)) : (void) relptr_store(base, (hp).relptr, value))
#define HnswPtrIsNull(base, hp) ((base) == NULL ? (hp).ptr == NULL : relptr_is_null((hp).relptr))
#define HnswPtrEqual(base, hp1, hp2) ((base) == NULL ? (hp1).ptr == (hp2).ptr : relptr_offset((hp1).relptr) == relptr_offset((hp2).relptr))

/* For code paths dedicated to each type */
#define HnswPtrPointer(hp) (hp).ptr
#define HnswPtrOffset(hp) relptr_offset((hp).relptr)

/* Variables */
extern int	hnsw_ef_search;
extern int	hnsw_iterative_scan;
extern int	hnsw_max_scan_tuples;
extern double hnsw_scan_mem_multiplier;
extern int	hnsw_lock_tranche_id;

/* I/O statistics tracking */
#define HNSW_IO_MAX_TRACKED_PAGES 512

/* Per-layer read statistics */
#define HNSW_MAX_TRACKED_LEVELS 256

typedef struct HnswIOPageSet
{
	int			npages;
	BlockNumber pages[HNSW_IO_MAX_TRACKED_PAGES];
	bool		overflow;
} HnswIOPageSet;

extern bool	hnsw_io_track;
extern int64 hnsw_io_read_pages;
extern int64 hnsw_io_write_pages;
extern int64 hnsw_io_total_ios;
extern int64 hnsw_io_updated_pages;
extern HnswIOPageSet *hnsw_io_page_set;
extern int	hnsw_io_current_layer;
extern int64 hnsw_io_layer_buffer_reads[HNSW_MAX_TRACKED_LEVELS];
extern int64 hnsw_io_layer_io_reads[HNSW_MAX_TRACKED_LEVELS];
extern bool	hnsw_io_layer_track;
/*
 * Per-session sequential/random read statistics.
 *
 * IMPORTANT: hnsw_io_last_block MUST ONLY be modified inside
 * HnswIOCountSeqRandRead().  External code must never write to
 * this variable directly -- the GUC guard (Rule 6) and seq/rand
 * classification logic depend on single-writer discipline.
 */
extern BlockNumber hnsw_io_last_block;
extern bool		hnsw_io_seq_rand_track;
extern int64	hnsw_io_seq_buffer_reads;
extern int64	hnsw_io_seq_io_reads;
extern int64	hnsw_io_rand_buffer_reads;
extern int64	hnsw_io_rand_io_reads;
extern int64	hnsw_io_first_buffer_reads;
extern int64	hnsw_io_first_io_reads;
extern void	HnswIOStatInit(void);
extern bool HnswIOMarkPageUpdated(BlockNumber blkno);

/* Write reason classification for detailed write amplification analysis */
typedef enum HnswWriteReason
{
	HNSW_WRITE_ADD_ELEMENT,				/* Write new element + neighbor tuples */
	HNSW_WRITE_UPDATE_NEIGHBOR,			/* Update existing neighbor's connection list */
	HNSW_WRITE_UPDATE_META_INSERT_PAGE,	/* Update metapage: insert page changed */
	HNSW_WRITE_UPDATE_META_ENTRY_POINT,	/* Update metapage: entry point upgraded */
	HNSW_WRITE_ADD_DUPLICATE,			/* Append heap TID to existing element */
	HNSW_WRITE_NREASONS					/* Must be last — array size sentinel */
} HnswWriteReason;

extern const char *hnsw_io_write_reason_names[];
extern int64 hnsw_io_write_reason_counts[HNSW_WRITE_NREASONS];

typedef enum HnswIterativeScanMode
{
	HNSW_ITERATIVE_SCAN_OFF,
	HNSW_ITERATIVE_SCAN_RELAXED,
	HNSW_ITERATIVE_SCAN_STRICT
}			HnswIterativeScanMode;

typedef struct HnswElementData HnswElementData;
typedef struct HnswNeighborArray HnswNeighborArray;

#define HnswPtrDeclare(type, relptrtype, ptrtype) \
	relptr_declare(type, relptrtype); \
	typedef union { type *ptr; relptrtype relptr; } ptrtype;

/* Pointers that can be absolute or relative */
/* Use char for DatumPtr so works with Pointer */
HnswPtrDeclare(HnswElementData, HnswElementRelptr, HnswElementPtr);
HnswPtrDeclare(HnswNeighborArray, HnswNeighborArrayRelptr, HnswNeighborArrayPtr);
HnswPtrDeclare(HnswNeighborArrayPtr, HnswNeighborsRelptr, HnswNeighborsPtr);
HnswPtrDeclare(char, DatumRelptr, DatumPtr);

struct HnswElementData
{
	HnswElementPtr next;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	uint8		heaptidsLength;
	uint8		level;
	uint8		deleted;
	uint8		version;
	uint32		hash;
	HnswNeighborsPtr neighbors;
	BlockNumber blkno;
	OffsetNumber offno;
	OffsetNumber neighborOffno;
	BlockNumber neighborPage;
	DatumPtr	value;
	LWLock		lock;

	/* AFTP build-time fields (valid only during FlushPages, not persisted to disk) */
	uint32		aftp_tuple_id;		/* ElementTuple sequential ID for AFTP Phase 1-4 */
	uint32		aftp_ntuple_id;		/* Corresponding NeighborTuple sequential ID */
};

typedef HnswElementData * HnswElement;

typedef struct HnswCandidate
{
	HnswElementPtr element;
	float		distance;
	bool		closer;
}			HnswCandidate;

struct HnswNeighborArray
{
	int			length;
	bool		closerSet;
	HnswCandidate items[FLEXIBLE_ARRAY_MEMBER];
};

typedef struct HnswSearchCandidate
{
	pairingheap_node c_node;
	pairingheap_node w_node;
	HnswElementPtr element;
	double		distance;
}			HnswSearchCandidate;

/* HNSW index options */
typedef struct HnswOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* number of connections */
	int			efConstruction; /* size of dynamic candidate list */
}			HnswOptions;

typedef struct HnswGraph
{
	/* Graph state */
	slock_t		lock;
	HnswElementPtr head;
	double		indtuples;

	/* Entry state */
	LWLock		entryLock;
	LWLock		entryWaitLock;
	HnswElementPtr entryPoint;

	/* Allocations state */
	LWLock		allocatorLock;
	Size		memoryUsed;
	Size		memoryTotal;

	/* Flushed state */
	LWLock		flushLock;
	bool		flushed;
}			HnswGraph;

typedef struct HnswShared
{
	/* Immutable state */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;

	/* Worker progress */
	ConditionVariable workersdonecv;

	/* Mutex for mutable state */
	slock_t		mutex;

	/* Mutable state */
	int			nparticipantsdone;
	double		reltuples;
	HnswGraph	graphData;
}			HnswShared;

#define ParallelTableScanFromHnswShared(shared) \
	(ParallelTableScanDesc) ((char *) (shared) + BUFFERALIGN(sizeof(HnswShared)))

typedef struct HnswLeader
{
	ParallelContext *pcxt;
	int			nparticipanttuplesorts;
	HnswShared *hnswshared;
	Snapshot	snapshot;
	char	   *hnswarea;
}			HnswLeader;

typedef struct HnswAllocator
{
	void	   *(*alloc) (Size size, void *state);
	void	   *state;
}			HnswAllocator;

typedef struct HnswTypeInfo
{
	int			maxDimensions;
	Datum		(*normalize) (PG_FUNCTION_ARGS);
	void		(*checkValue) (Pointer v);
}			HnswTypeInfo;

typedef struct HnswSupport
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
}			HnswSupport;

typedef struct HnswQuery
{
	Datum		value;
}			HnswQuery;

/* Forward declaration needed by hnswaftp.h (included below) */
typedef struct HnswBuildState HnswBuildState;
#include "hnswaftp.h"

typedef struct HnswBuildState
{
	/* Info */
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;
	const		HnswTypeInfo *typeInfo;

	/* Settings */
	int			dimensions;
	int			m;
	int			efConstruction;

	/* Statistics */
	double		indtuples;
	double		reltuples;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	HnswGraph	graphData;
	HnswGraph  *graph;
	double		ml;
	int			maxLevel;

	/* Memory */
	MemoryContext graphCtx;
	MemoryContext tmpCtx;
	HnswAllocator allocator;

	/* Parallel builds */
	HnswLeader *hnswleader;
	HnswShared *hnswshared;
	char	   *hnswarea;

	/* AFTP packing results (NULL when AFTP is disabled) */
	int	   *aftp_etuple_page;	/* P_E[t] = page_id, length = n_elements */
	int	   *aftp_ntuple_page;	/* P_N[t] = page_id, length = n_elements */
	int	   *aftp_etuple_order;	/* Sorted ElementTuple write order (by page_id, tuple_id) */
	int	   *aftp_ntuple_order;	/* Sorted NeighborTuple write order */
	int		   aftp_n_epages;		/* Number of ElementTuple pages */
	int		   aftp_n_npages;		/* Number of NeighborTuple pages */

	/* AFTP memory context and lifecycle management */
	MemoryContext aftp_context;		/* Dedicated MemoryContext for all AFTP allocations */
	AftpBuildStatus aftp_status;	/* AFTP build state machine */
}			HnswBuildState;

typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	uint16		m;
	uint16		efConstruction;
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int16		entryLevel;
	BlockNumber insertPage;
	BlockNumber neighborInsertPage;

	/* AFTP fields (valid when storage_version >= 3, zero for non-AFTP layouts) */
	uint8		aftp_build_status;		/* AftpBuildStatus */
	uint8		aftp_storage_version;		/* AFTP layout format version (current = 1) */
	uint16		aftp_n_epages;			/* Number of ElementTuple optimized pages (0 = not optimized) */
	uint16		aftp_n_npages;			/* Number of NeighborTuple optimized pages (0 = not optimized) */
}			HnswMetaPageData;

typedef HnswMetaPageData * HnswMetaPage;

typedef struct HnswPageOpaqueData
{
	BlockNumber nextblkno;
	uint16		unused;
	uint16		page_id;		/* for identification of HNSW indexes */
}			HnswPageOpaqueData;

typedef HnswPageOpaqueData * HnswPageOpaque;

typedef struct HnswElementTupleData
{
	uint8		type;
	uint8		level;
	uint8		deleted;
	uint8		version;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	ItemPointerData neighbortid;
	uint16		unused;
	Vector		data;
}			HnswElementTupleData;

typedef HnswElementTupleData * HnswElementTuple;

typedef struct HnswNeighborTupleData
{
	uint8		type;
	uint8		version;
	uint16		count;
	ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
}			HnswNeighborTupleData;

typedef HnswNeighborTupleData * HnswNeighborTuple;

typedef union
{
	struct pointerhash_hash *pointers;
	struct offsethash_hash *offsets;
	struct tidhash_hash *tids;
}			visited_hash;

typedef union
{
	HnswElement element;
	ItemPointerData indextid;
}			HnswUnvisited;

typedef struct HnswScanOpaqueData
{
	const		HnswTypeInfo *typeInfo;
	bool		first;
	List	   *w;
	visited_hash v;
	pairingheap *discarded;
	HnswQuery	q;
	int			m;
	int64		tuples;
	double		previousDistance;
	Size		maxMemory;
	MemoryContext tmpCtx;

	/* Support functions */
	HnswSupport support;
}			HnswScanOpaqueData;

typedef HnswScanOpaqueData * HnswScanOpaque;

typedef struct HnswVacuumState
{
	/* Info */
	Relation	index;
	IndexBulkDeleteResult *stats;
	IndexBulkDeleteCallback callback;
	void	   *callback_state;

	/* Settings */
	int			m;
	int			efConstruction;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	struct tidhash_hash *deleted;
	BufferAccessStrategy bas;
	HnswNeighborTuple ntup;
	HnswElementData highestPoint;

	/* Memory */
	MemoryContext tmpCtx;
}			HnswVacuumState;

/* Methods */
int			HnswGetM(Relation index);
int			HnswGetEfConstruction(Relation index);
FmgrInfo   *HnswOptionalProcInfo(Relation index, uint16 procnum);
void		HnswInitSupport(HnswSupport * support, Relation index);
Datum		HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value);
bool		HnswCheckNorm(HnswSupport * support, Datum value);
Buffer		HnswNewBuffer(Relation index, ForkNumber forkNum);
void		HnswInitPage(Buffer buf, Page page);
void		HnswInit(void);
List	   *HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples);
HnswElement HnswGetEntryPoint(Relation index);
void		HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint);
void	   *HnswAlloc(HnswAllocator * allocator, Size size);
HnswElement HnswInitElement(char *base, ItemPointer tid, int m, double ml, int maxLevel, HnswAllocator * alloc);
HnswElement HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno);
void		HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing);
HnswSearchCandidate *HnswEntryCandidate(char *base, HnswElement em, HnswQuery * q, Relation rel, HnswSupport * support, bool loadVec);
void		HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, BlockNumber neighborInsertPage, ForkNumber forkNum, bool building, HnswWriteReason reason);
void		HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m);
void		HnswAddHeapTid(HnswElement element, ItemPointer heaptid);
HnswNeighborArray *HnswInitNeighborArray(int lm, HnswAllocator * allocator);
void		HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * alloc);
bool		HnswInsertTupleOnDisk(Relation index, HnswSupport * support, Datum value, ItemPointer heaptid, bool building);
void		HnswUpdateNeighborsOnDisk(Relation index, HnswSupport * support, HnswElement e, int m, bool checkExisting, bool building);
void		HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec);
void		HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance);
bool		HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support);
void		HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element);
void		HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support);
bool		HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc);
void		HnswInitLockTranche(void);
const		HnswTypeInfo *HnswGetTypeInfo(Relation index);
PGDLLEXPORT void HnswParallelBuildMain(dsm_segment *seg, shm_toc *toc);

/* Index access methods */
IndexBuildResult *hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		hnswbuildempty(Relation index);
bool		hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
IndexBulkDeleteResult *hnswbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *hnswvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc hnswbeginscan(Relation index, int nkeys, int norderbys);
void		hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool		hnswgettuple(IndexScanDesc scan, ScanDirection dir);
void		hnswendscan(IndexScanDesc scan);

static inline HnswNeighborArray *
HnswGetNeighbors(char *base, HnswElement element, int lc)
{
	HnswNeighborArrayPtr *neighborList = HnswPtrAccess(base, element->neighbors);

	Assert(element->level >= lc);

	return HnswPtrAccess(base, neighborList[lc]);
}

/* Hash tables */
typedef struct TidHashEntry
{
	ItemPointerData tid;
	char		status;
}			TidHashEntry;

#define SH_PREFIX tidhash
#define SH_ELEMENT_TYPE TidHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct PointerHashEntry
{
	uintptr_t	ptr;
	char		status;
}			PointerHashEntry;

#define SH_PREFIX pointerhash
#define SH_ELEMENT_TYPE PointerHashEntry
#define SH_KEY_TYPE uintptr_t
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct OffsetHashEntry
{
	Size		offset;
	char		status;
}			OffsetHashEntry;

#define SH_PREFIX offsethash
#define SH_ELEMENT_TYPE OffsetHashEntry
#define SH_KEY_TYPE Size
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

/*
 * HnswIOCountLayerReadBegin / HnswIOCountLayerReadEnd
 *
 * Pair of helpers for per-layer read I/O statistics.
 * Must be called around every ReadBuffer() site that participates
 * in graph-traversal reads:
 *
 *   int64 io_before = 0;
 *   if (hnsw_io_track)
 *       HnswIOCountLayerReadBegin(&io_before);
 *   buf = ReadBuffer(index, blkno);   -- EXACTLY ONE ReadBuffer between Begin/End
 *   if (hnsw_io_track)
 *       HnswIOCountLayerReadEnd(io_before, blkno);
 *
 *   CONSTRAINT: No other ReadBuffer (or any operation that may
 *   increment pgBufferUsage.shared_blks_read) must occur between
 *   Begin and End.  The io_before snapshot is only valid for the
 *   immediately following ReadBuffer call.
 *
 *   Even when hnsw_io_track is false, io_before is declared (and
 *   initialised to 0) so that the variable is in scope for the
 *   End call.  The compiler will eliminate the dead stores.
 *
 * IO detection: snapshots pgBufferUsage.shared_blks_read before
 * ReadBuffer() and compares after.  If the counter incremented,
 * a physical disk read occurred.
 *
 * The caller must set hnsw_io_current_layer to the active search
 * layer (>= 0) before the ReadBuffer.  Layer -1 means "no layer
 * context" (e.g. metapage reads) and skips per-layer counting.
 */
static inline void
HnswIOCountLayerReadBegin(int64 *io_before)
{
#if PG_VERSION_NUM >= 140000
	*io_before = pgBufferUsage.shared_blks_read;
#else
	/*
	 * PG <= 13: pgBufferUsage is a plain PgStat_Counter, not a struct.
	 * Fall back to counting every ReadBuffer as a physical read (safe
	 * over-count; the io_reads <= buffer_reads invariant still holds).
	 */
	*io_before = 0;
#endif
}

/*
 * Classification helper for sequential/random/first read detection.
 * Called inside HnswIOCountLayerReadEnd so every ReadBuffer() site
 * that uses the Begin/End pattern gets seq/rand stats automatically.
 */
static inline void
HnswIOCountSeqRandRead(BlockNumber blkno, bool is_physical_io)
{
	/* Gated by independent GUC */
	if (!hnsw_io_seq_rand_track)
		return;

	/* First read: no predecessor block */
	if (hnsw_io_last_block == InvalidBlockNumber)
	{
		hnsw_io_first_buffer_reads++;
		if (is_physical_io)
			hnsw_io_first_io_reads++;
	}
	/* Sequential: consecutive ascending or descending
	 * (SSD performance is symmetric for forward/reverse) */
	else if (blkno == hnsw_io_last_block + 1 ||
			 blkno == hnsw_io_last_block - 1)
	{
		hnsw_io_seq_buffer_reads++;
		if (is_physical_io)
			hnsw_io_seq_io_reads++;
	}
	/* Random: everything else */
	else
	{
		hnsw_io_rand_buffer_reads++;
		if (is_physical_io)
			hnsw_io_rand_io_reads++;
	}

	/* Track predecessor regardless of classification */
	hnsw_io_last_block = blkno;
}

static inline void
HnswIOCountLayerReadEnd(int64 io_before, BlockNumber blkno)
{
	bool		is_physical_io = false;

	/* Global counters always updated (backward compatible) */
	hnsw_io_read_pages++;
	hnsw_io_total_ios++;

#if PG_VERSION_NUM >= 140000
	if (pgBufferUsage.shared_blks_read > io_before)
		is_physical_io = true;
#else
	/*
	 * PG <= 13 fallback: treat every ReadBuffer as physical I/O.
	 * This over-counts but preserves the invariant
	 * io_reads == buffer_reads (cache miss rate = 100%).
	 */
	is_physical_io = true;
#endif

	/* Sequential/random classification (gated by own GUC internally) */
	HnswIOCountSeqRandRead(blkno, is_physical_io);

	/* Per-layer tracking: gated by GUC + layer context */
	if (!hnsw_io_layer_track)
		return;

	if (hnsw_io_current_layer >= 0 &&
		hnsw_io_current_layer < HNSW_MAX_TRACKED_LEVELS)
	{
		hnsw_io_layer_buffer_reads[hnsw_io_current_layer]++;

		if (is_physical_io)
			hnsw_io_layer_io_reads[hnsw_io_current_layer]++;
	}
}
#endif
