# evalkit

Scripts that produce the numbers quoted in docstrings. They read the corpus on
the training host strictly read-only (`mode=ro`, `PRAGMA query_only=ON`) and
write nothing.

`against_old_serialiser.py N` — the header-directed read, this package against
`scans/serialize2.py`, N tables per source. Locate the axis header, resolve its
column, read that column down the table, compare against the coordinates the
extractor already found.

    source     tables |  old x-read    old 100% |  new x-read    new 100%   packed
    pubget         80 |       56.1%       47.4% |       70.0%       61.4%        7
    ace            80 |       57.6%       52.7% |       86.3%       83.7%       16
    elsevier       80 |       29.3%       10.2% |       60.4%       40.8%       18
    pdf            80 |        0.0%        0.0% |       93.4%       91.7%       27

The last column counts tables where no axis header exists but a column holds
whole coordinate triples, which the reader falls back to.
