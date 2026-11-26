# F5 mechanism check

## Arm M-U

verdicts: {'not_retired': 5, 'retired_by_N': 11}; controls with retired edges: 5/22

| repo | pair | old -> new | O edges | O expired | expired during N | verdict | via |
|---|---|---|---:|---:|---:|---|---|
| digitalservicebund/digitalcheck-dito | dito-P1 | 0003 -> 0027 | 9 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P2 | 0011,0025 -> 0028 | 24 | 6 | 4 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P3 | 0015 -> 0027 | 8 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P4 | 0016 -> 0027 | 5 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P5 | 0017 -> 0027 | 15 | 4 | 4 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P6 | 0022 -> 0027 | 18 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P7 | 0009 -> 0018 | 21 | 5 | 4 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P8 | 0003,0012 -> 0025 | 14 | 2 | 2 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P9 | 0007 -> 0019 | 12 | 5 | 5 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P10 | 0013 -> 0015 | 25 | 0 | 0 | not_retired |  |
| supabase/cli | supa-P11 | 0011 -> 0028 | 34 | 5 | 3 | retired_by_N | N_ingest |
| supabase/cli | supa-P12 | 0016 -> 0031 | 30 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P13 | 0004 -> 0024 | 28 | 2 | 1 | retired_by_N | N_ingest |
| FIWARE/data-space-connector | fiware-P14 | 0001 -> 0008 | 19 | 3 | 3 | retired_by_N | N_ingest |
| FIWARE/data-space-connector | fiware-P15 | 0001 -> 0008 | 19 | 3 | 3 | retired_by_N | N_ingest |
| phel-lang/phel-lang | phel-P16 | 0005 -> 0021 | 17 | 2 | 2 | retired_by_N | N_ingest |

| repo | control | edges | expired | expired during |
|---|---|---:|---:|---|
| digitalservicebund/digitalcheck-dito | 0002 | 17 | 1 | 0003 |
| digitalservicebund/digitalcheck-dito | 0024 | 17 | 0 |  |
| digitalservicebund/digitalcheck-dito | 0026 | 13 | 0 |  |
| digitalservicebund/ris-search | 0002 | 8 | 0 |  |
| digitalservicebund/ris-search | 0004 | 18 | 1 | 0018 |
| digitalservicebund/ris-search | 0011 | 25 | 2 | 0018 |
| digitalservicebund/ris-search | 0014 | 11 | 0 |  |
| digitalservicebund/ris-search | 0017 | 14 | 0 |  |
| digitalservicebund/ris-search | 0020 | 11 | 0 |  |
| digitalservicebund/ris-search | 0021 | 10 | 1 | 0021 |
| digitalservicebund/ris-search | 0022 | 11 | 1 | 0022 |
| digitalservicebund/ris-search | 0024 | 6 | 0 |  |
| supabase/cli | 0013-live | 28 | 0 |  |
| supabase/cli | 0014 | 31 | 0 |  |
| supabase/cli | 0034 | 9 | 0 |  |
| supabase/cli | 0030 | 22 | 0 |  |
| FIWARE/data-space-connector | 0006 | 31 | 0 |  |
| FIWARE/data-space-connector | 0007 | 20 | 0 |  |
| phel-lang/phel-lang | 0002 | 30 | 0 |  |
| phel-lang/phel-lang | 0003 | 30 | 0 |  |
| phel-lang/phel-lang | 0011 | 31 | 0 |  |
| phel-lang/phel-lang | 0012 | 27 | 0 |  |
