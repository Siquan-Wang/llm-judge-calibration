# Pinned author-reference numerical comparison

Reference: `ppi-python==0.2.3`. Cases: 6; comparisons: 24. All numerical convention checks passed: **True**.

Maximum fixed-power point difference: 5.55e-17. Maximum automatic-power point difference: 0.0106383; maximum automatic coefficient difference: 0.0516802.

Fixed-power points should match. Interval widths differ because judgecal uses ddof=1 per-pool variance and this reference uses ddof=0 interval moments. Automatic coefficients also differ: judgecal uses separate-pool ddof=1 moments, while the reference uses audit covariance with divisor n and pooled judge variance with ddof=1. Every observed difference is retained in comparisons.csv. The reference coefficient column is computed from the pinned source formula and checked through its public point/CI calls, not returned by the API.

This validates numerical conventions on the exported fixtures; it does not establish statistical superiority, real-data accuracy, population coverage, or clustered-reference equivalence. The script performed no downloads or model calls. See cases.json and environment.json for exact arrays, source fingerprints, API signatures, and installed versions.
