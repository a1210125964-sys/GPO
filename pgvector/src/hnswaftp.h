#ifndef HNSWAFTP_H
#define HNSWAFTP_H

#include "postgres.h"
#include "utils/guc.h"
#include "hnsw.h"

/* ----------------------------------------------------------------
 * AFTP (Access Frequency-driven Tuple Packing) Data Structures
 *
 * Build-time optimization that reorders tuple placement within
 * pages to improve buffer hit rate during HNSW search.
 *
 * Affinity is computed between ElementTuples only; NeighborTuple
 * trace entries are position markers.  Packing groups whole
 * elements, and each element's ElementTuple and NeighborTuple are
 * always written to the same page, sized together (both tuples
 * plus two line pointers).
 *
 * Four-phase algorithm:
 *   Phase 1 — Trace Collection
 *   Phase 2 — Affinity Matrix
 *   Phase 3 — Greedy Packing
 *   Phase 4 — Iterative Refinement
 * ----------------------------------------------------------------
 */

/* ========================
 * 1. Trace Collection (§3.2)
 * ======================== */

/* AFTP trace types */
#define AFTP_TUPLE_ELEMENT  1
#define AFTP_TUPLE_NEIGHBOR 2

/* Single trace entry — records one tuple access during search */
typedef struct AftpTraceEntry
{
	uint32		tuple_id;		/* Sequential ID assigned during build */
	uint8		type;			/* AFTP_TUPLE_ELEMENT or AFTP_TUPLE_NEIGHBOR */
	uint8		layer;			/* HNSW layer where access occurred */
	uint16		padding;		/* Alignment padding */
}			AftpTraceEntry;

/* Trace for a single sample query — dynamically allocated array */
typedef struct AftpQueryTrace
{
	int			nentries;		/* Number of entries recorded */
	int			max_entries;	/* Allocated capacity */
	AftpTraceEntry *entries;	/* Dynamic array of trace entries */
}			AftpQueryTrace;

/* Global trace collection state */
typedef struct AftpTraceState
{
	int			n_queries;		/* Target sample count (aftp_sample_queries) */
	int			n_completed;	/* Actually completed samples */
	AftpQueryTrace *traces;		/* Array of n_queries traces */
	int			n_elements;		/* Total ElementTuple count */
	int			n_neighbors;	/* Total NeighborTuple count */
}			AftpTraceState;

/* ===============================
 * 2. Affinity Matrix Structures (§3.3)
 * =============================== */

/* Single affinity entry — one (target, weight) pair */
typedef struct AftpAffinityEntry
{
	uint32		tuple_id;		/* Target tuple ID */
	float		affinity;		/* A(source, target) value */
}			AftpAffinityEntry;

/* Sparse row for one tuple — top-K affinity entries sorted descending */
typedef struct AftpAffinityRow
{
	uint32		tuple_id;		/* This tuple's ID */
	uint32		n_entries;		/* Number of non-zero entries (<= max_entries) */
	uint32		max_entries;	/* Allocated capacity */
	AftpAffinityEntry *entries;	/* Array, sorted by affinity descending */
}			AftpAffinityRow;

/* Complete sparse affinity matrix */
typedef struct AftpAffinityMatrix
{
	uint32		n_tuples;		/* Total tuples */
	uint32		max_entries;	/* K_affinity = aftp_top_affinity */
	AftpAffinityRow *rows;		/* One row per tuple */
	float	   *point_weights;	/* Cached total affinity per tuple */
}			AftpAffinityMatrix;

/* ============================
 * 3. Packing Structures (§3.4, §3.5)
 * ============================ */

/* One packed group — written as one physical PostgreSQL page holding each
 * element's ElementTuple and NeighborTuple */
typedef struct AftpPage
{
	int			page_id;		/* Logical page ID */
	int			n_tuples;		/* Current number of elements in this page */
	int			tuple_capacity;	/* Allocated entries in tuple_ids */
	Size		used_bytes;		/* Bytes of both tuples plus line pointers */
	Size		capacity_bytes;	/* Exact usable bytes in a PostgreSQL page */
	uint32	   *tuple_ids;		/* Array of element IDs in this page */
}			AftpPage;

/* Page→element inverted index for Phase 4 fast candidate filtering (§3.5) */
typedef struct AftpInvertedIndex
{
	uint32	  **page_partners;	/* page_partners[p] = sorted array of external element IDs
								 * that have non-zero affinity with page p */
	uint32	   *page_npartners; /* Number of partners per page */
	int			n_pages;		/* Total page count */
}			AftpInvertedIndex;

/* =================================
 * 4. Backfill Structure (§4.5)
 * ================================= */

/* Entry for the third-pass backfill operation */
typedef struct AftpBackfillEntry
{
	OffsetNumber offno;			/* ElementTuple offset within its page */
	BlockNumber neighborPage;	/* NeighborTuple page (determined in pass 2) */
	OffsetNumber neighborOffno; /* NeighborTuple offset (determined in pass 2) */
}			AftpBackfillEntry;

/* =================================
 * 5. Build Status Enum (§4.3)
 * ================================= */

/* AFTP build status — persisted in metapage for crash recovery */
typedef enum AftpBuildStatus
{
	AFTP_BUILD_NOT_STARTED = 0, /* AFTP not enabled or not yet started */
	AFTP_BUILD_IN_PROGRESS = 1, /* Phase 1-4 in progress */
	AFTP_BUILD_FLUSHING = 2,	/* Three-pass write in progress */
	AFTP_BUILD_COMPLETED = 3,	/* AFTP build successfully completed */
}			AftpBuildStatus;

/* =================================
 * 6. AFTP Configuration (§4.4)
 * ================================= */

/* AFTP configuration — populated from GUC parameters */
typedef struct AftpConfig
{
	bool		enabled;
	int			sample_queries; /* S: default 200 */
	int			ef_search;		/* ef for trace collection: default 64 */
	int			window;			/* W: co-visit window: default 100 */
	int			top_affinity;	/* K_affinity: default 20 */
	int			refine_iterations;	/* beta: default 5, 0 = skip Phase 4 */
	double		decay_lambda;	/* lambda: 0 = auto (W/3) */
	int			seed_softmax;	/* lambda_seed: default 10, 0 = deterministic */
	int			max_memory_mb;	/* Hard memory limit: 0 = auto */
}			AftpConfig;

/* =================================
 * 7. GUC Variable Declarations (§4.4)
 * ================================= */

/* GUC variables — defined in hnswaftp.c, used by hnsw.c for registration */
extern bool		aftp_enable;
extern int		aftp_sample_queries;
extern int		aftp_ef_search;
extern int		aftp_window;
extern int		aftp_top_affinity;
extern int		aftp_refine_iterations;
extern double	aftp_decay_lambda;
extern int		aftp_seed_softmax;
extern int		aftp_max_memory;

/* =================================
 * 8. Algorithm API Declarations
 * ================================= */

/* Main AFTP entry point */
extern bool AftpMain(HnswBuildState *buildstate, AftpConfig *config);

/* Phase 1: Trace collection */
extern AftpTraceState *AftpCollectTraces(HnswBuildState *buildstate, AftpConfig *config);

/* Phase 2: Affinity matrix construction (ElementTuple pairs only; n_tuples
 * is the element count and IDs are element IDs) */
extern AftpAffinityMatrix *AftpBuildAffinityMatrix(AftpTraceState *traces,
	int n_tuples, int top_affinity, int window, double decay_lambda);

/* Phase 3: Greedy seed-and-grow packing */
extern void AftpGreedyPacking(AftpAffinityMatrix *A, Size *tuple_sizes,
	Size page_capacity, int seed_softmax,
	int **out_page_assignment, AftpPage ***out_pages, int *out_n_pages);

/* Phase 4: Iterative BNF-style refinement */
extern void AftpIterativeRefinement(AftpAffinityMatrix *A, int *P, AftpPage **pages,
	int n_pages, int n_tuples, int max_iterations, Size *tuple_sizes);

/* Utility: populate AftpConfig from GUC variables */
extern void AftpPopulateConfig(AftpConfig *cfg);

/* Utility: assign sequential IDs to all tuples */
extern void AftpAssignTupleIds(HnswBuildState *buildstate, int *n_elements, int *n_neighbors);

/* Utility: compute locality score LS(P) */
extern float AftpComputeLocalityScore(AftpAffinityMatrix *A, int *P, int n_tuples);

/* Utility: check trace convergence between two halves */
extern bool AftpCheckTraceConvergence(AftpQueryTrace *traces, int n_queries,
	int n_tuples, int top_affinity);

/* Utility: free resources */
extern void AftpFreeTraceState(AftpTraceState *state);
extern void AftpFreeAffinityMatrix(AftpAffinityMatrix *A);

#endif							/* HNSWAFTP_H */
