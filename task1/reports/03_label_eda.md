# Label EDA — train & validation

## train

- Documents: **90**, tokens: **95,138**, watermarked: **44,946 (47.2%)**
- Docs without any watermark span: **2**
- Spans starting at doc position 0: **26**, spans ending at doc end: **18** (of 396 spans)

| distribution | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| doc length (tokens) | 90 | 538 | 869 | 1058 | 1251 | 1528 |
| span length (tokens) | 396 | 24 | 47 | 95 | 159 | 320 |
| gap between spans | 308 | 12 | 22 | 68 | 135 | 608 |
| spans per doc | 90 | 0 | 4 | 5 | 5 | 6 |

Spans per doc histogram: 0 spans: 2, 1 spans: 2, 2 spans: 4, 3 spans: 12, 4 spans: 22, 5 spans: 26, 6 spans: 22

Relative span start positions (fraction of doc): 5%=0.00, 25%=0.16, 50%=0.41, 75%=0.66, 95%=0.92

## validation

- Documents: **90**, tokens: **95,704**, watermarked: **44,325 (46.3%)**
- Docs without any watermark span: **2**
- Spans starting at doc position 0: **26**, spans ending at doc end: **18** (of 396 spans)

| distribution | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| doc length (tokens) | 90 | 542 | 839 | 1038 | 1250 | 1527 |
| span length (tokens) | 396 | 31 | 47 | 95 | 159 | 320 |
| gap between spans | 308 | 12 | 23 | 60 | 149 | 660 |
| spans per doc | 90 | 0 | 4 | 5 | 5 | 6 |

Spans per doc histogram: 0 spans: 2, 1 spans: 2, 2 spans: 4, 3 spans: 12, 4 spans: 22, 5 spans: 26, 6 spans: 22

Relative span start positions (fraction of doc): 5%=0.00, 25%=0.19, 50%=0.45, 75%=0.70, 95%=0.88
