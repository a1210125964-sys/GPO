# Deep100K

## Data source

- Source: Deep1B official `base.1B.fbin`
- URL: https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
- Downloaded byte range: `0-384000007` (first 1,000,000 vectors)
- Source prefix SHA-256: `1c5169a4eb8dd860727f47ca641e48f06a2253b7a235ecf4cbde7dda3ec893c3`
- Source format/header: little-endian uint32 count=1,000,000,000; uint32 dimension=96; float32 vectors
- Distance: angular
- Dimension: 96

## Deterministic sampling and split

- Sample: 100,000 vectors without replacement from the downloaded 1M-vector prefix
- Sampling seed: 20260827
- Build: 75,000 vectors in 3 Parquet files of 25,000 each
- Insert: 25,000 vectors in one Parquet file
- Vector column: `embedding` (float32 fixed-size list, length 96)
