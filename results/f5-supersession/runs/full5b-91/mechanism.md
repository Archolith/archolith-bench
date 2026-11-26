# F5 mechanism check

## Arm M-U

verdicts: {'retired_by_other': 1, 'retired_by_N': 12, 'not_retired': 3}; controls with retired edges: 1/22

| repo | pair | old -> new | O edges | O expired | expired during N | verdict | via |
|---|---|---|---:|---:|---:|---|---|
| digitalservicebund/digitalcheck-dito | dito-P1 | 0003 -> 0027 | 13 | 1 | 0 | retired_by_other |  |
| digitalservicebund/digitalcheck-dito | dito-P2 | 0011,0025 -> 0028 | 26 | 4 | 2 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P3 | 0015 -> 0027 | 9 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P4 | 0016 -> 0027 | 11 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P5 | 0017 -> 0027 | 17 | 1 | 1 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P6 | 0022 -> 0027 | 14 | 3 | 3 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P7 | 0009 -> 0018 | 22 | 5 | 3 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P8 | 0003,0012 -> 0025 | 20 | 0 | 0 | not_retired |  |
| digitalservicebund/ris-search | ris-P9 | 0007 -> 0019 | 10 | 7 | 7 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P10 | 0013 -> 0015 | 24 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P11 | 0011 -> 0028 | 54 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P12 | 0016 -> 0031 | 19 | 2 | 2 | retired_by_N | N_ingest |
| supabase/cli | supa-P13 | 0004 -> 0024 | 44 | 7 | 2 | retired_by_N | N_ingest |
| FIWARE/data-space-connector | fiware-P14 | 0001 -> 0008 | 20 | 2 | 2 | retired_by_N | N_ingest |
| FIWARE/data-space-connector | fiware-P15 | 0001 -> 0008 | 20 | 2 | 2 | retired_by_N | N_ingest |
| phel-lang/phel-lang | phel-P16 | 0005 -> 0021 | 22 | 4 | 4 | retired_by_N | N_ingest |

| repo | control | edges | expired | expired during |
|---|---|---:|---:|---|
| digitalservicebund/digitalcheck-dito | 0002 | 10 | 0 |  |
| digitalservicebund/digitalcheck-dito | 0024 | 17 | 0 |  |
| digitalservicebund/digitalcheck-dito | 0026 | 15 | 0 |  |
| digitalservicebund/ris-search | 0002 | 9 | 0 |  |
| digitalservicebund/ris-search | 0004 | 21 | 0 |  |
| digitalservicebund/ris-search | 0011 | 22 | 2 | 0018 |
| digitalservicebund/ris-search | 0014 | 16 | 0 |  |
| digitalservicebund/ris-search | 0017 | 18 | 0 |  |
| digitalservicebund/ris-search | 0020 | 13 | 0 |  |
| digitalservicebund/ris-search | 0021 | 10 | 0 |  |
| digitalservicebund/ris-search | 0022 | 12 | 0 |  |
| digitalservicebund/ris-search | 0024 | 5 | 0 |  |
| supabase/cli | 0013-live | 22 | 0 |  |
| supabase/cli | 0014 | 51 | 0 |  |
| supabase/cli | 0034 | 11 | 0 |  |
| supabase/cli | 0030 | 23 | 0 |  |
| FIWARE/data-space-connector | 0006 | 31 | 0 |  |
| FIWARE/data-space-connector | 0007 | 19 | 0 |  |
| phel-lang/phel-lang | 0002 | 27 | 0 |  |
| phel-lang/phel-lang | 0003 | 29 | 0 |  |
| phel-lang/phel-lang | 0011 | 42 | 0 |  |
| phel-lang/phel-lang | 0012 | 26 | 0 |  |
