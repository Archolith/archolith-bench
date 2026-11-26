# F5 mechanism check

## Arm M-U

verdicts: {'not_retired': 2, 'retired_by_N': 14}; controls with retired edges: 3/22

| repo | pair | old -> new | O edges | O expired | expired during N | verdict | via |
|---|---|---|---:|---:|---:|---|---|
| digitalservicebund/digitalcheck-dito | dito-P1 | 0003 -> 0027 | 7 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P2 | 0011,0025 -> 0028 | 19 | 5 | 3 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P3 | 0015 -> 0027 | 10 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P4 | 0016 -> 0027 | 5 | 1 | 1 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P5 | 0017 -> 0027 | 14 | 4 | 4 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P6 | 0022 -> 0027 | 13 | 2 | 2 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P7 | 0009 -> 0018 | 23 | 4 | 3 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P8 | 0003,0012 -> 0025 | 13 | 1 | 1 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P9 | 0007 -> 0019 | 9 | 6 | 6 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P10 | 0013 -> 0015 | 31 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P11 | 0011 -> 0028 | 43 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P12 | 0016 -> 0031 | 18 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P13 | 0004 -> 0024 | 45 | 4 | 2 | retired_by_N | N_ingest |
| FIWARE/data-space-connector | fiware-P14 | 0001 -> 0008 | 19 | 3 | 3 | retired_by_N | N_ingest |
| FIWARE/data-space-connector | fiware-P15 | 0001 -> 0008 | 19 | 3 | 3 | retired_by_N | N_ingest |
| phel-lang/phel-lang | phel-P16 | 0005 -> 0021 | 24 | 2 | 2 | retired_by_N | N_ingest |

| repo | control | edges | expired | expired during |
|---|---|---:|---:|---|
| digitalservicebund/digitalcheck-dito | 0002 | 11 | 1 | 0018 |
| digitalservicebund/digitalcheck-dito | 0024 | 18 | 0 |  |
| digitalservicebund/digitalcheck-dito | 0026 | 7 | 0 |  |
| digitalservicebund/ris-search | 0002 | 6 | 0 |  |
| digitalservicebund/ris-search | 0004 | 20 | 0 |  |
| digitalservicebund/ris-search | 0011 | 22 | 0 |  |
| digitalservicebund/ris-search | 0014 | 21 | 3 | 0014 |
| digitalservicebund/ris-search | 0017 | 14 | 0 |  |
| digitalservicebund/ris-search | 0020 | 11 | 0 |  |
| digitalservicebund/ris-search | 0021 | 13 | 1 | 0021 |
| digitalservicebund/ris-search | 0022 | 12 | 0 |  |
| digitalservicebund/ris-search | 0024 | 6 | 0 |  |
| supabase/cli | 0013-live | 48 | 0 |  |
| supabase/cli | 0014 | 43 | 0 |  |
| supabase/cli | 0034 | 12 | 0 |  |
| supabase/cli | 0030 | 21 | 0 |  |
| FIWARE/data-space-connector | 0006 | 32 | 0 |  |
| FIWARE/data-space-connector | 0007 | 25 | 0 |  |
| phel-lang/phel-lang | 0002 | 29 | 0 |  |
| phel-lang/phel-lang | 0003 | 38 | 0 |  |
| phel-lang/phel-lang | 0011 | 32 | 0 |  |
| phel-lang/phel-lang | 0012 | 23 | 0 |  |
