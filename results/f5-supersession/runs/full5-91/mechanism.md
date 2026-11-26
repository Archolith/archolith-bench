# F5 mechanism check

## Arm M-U

verdicts: {'not_retired': 3, 'retired_by_N': 13}; controls with retired edges: 2/22

| repo | pair | old -> new | O edges | O expired | expired during N | verdict | via |
|---|---|---|---:|---:|---:|---|---|
| digitalservicebund/digitalcheck-dito | dito-P1 | 0003 -> 0027 | 9 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P2 | 0011,0025 -> 0028 | 22 | 2 | 1 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P3 | 0015 -> 0027 | 10 | 0 | 0 | not_retired |  |
| digitalservicebund/digitalcheck-dito | dito-P4 | 0016 -> 0027 | 2 | 1 | 1 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P5 | 0017 -> 0027 | 13 | 2 | 2 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P6 | 0022 -> 0027 | 13 | 3 | 3 | retired_by_N | N_ingest |
| digitalservicebund/digitalcheck-dito | dito-P7 | 0009 -> 0018 | 19 | 2 | 1 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P8 | 0003,0012 -> 0025 | 14 | 0 | 0 | not_retired |  |
| digitalservicebund/ris-search | ris-P9 | 0007 -> 0019 | 12 | 6 | 6 | retired_by_N | N_ingest |
| digitalservicebund/ris-search | ris-P10 | 0013 -> 0015 | 22 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P11 | 0011 -> 0028 | 26 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P12 | 0016 -> 0031 | 11 | 1 | 1 | retired_by_N | N_ingest |
| supabase/cli | supa-P13 | 0004 -> 0024 | 35 | 4 | 1 | retired_by_N | N_ingest |
| FIWARE/data-space-connector | fiware-P14 | 0001 -> 0008 | 28 | 3 | 3 | retired_by_N | N_ingest |
| FIWARE/data-space-connector | fiware-P15 | 0001 -> 0008 | 28 | 3 | 3 | retired_by_N | N_ingest |
| phel-lang/phel-lang | phel-P16 | 0005 -> 0021 | 22 | 3 | 3 | retired_by_N | N_ingest |

| repo | control | edges | expired | expired during |
|---|---|---:|---:|---|
| digitalservicebund/digitalcheck-dito | 0002 | 9 | 1 | 0018 |
| digitalservicebund/digitalcheck-dito | 0024 | 19 | 0 |  |
| digitalservicebund/digitalcheck-dito | 0026 | 9 | 0 |  |
| digitalservicebund/ris-search | 0002 | 11 | 0 |  |
| digitalservicebund/ris-search | 0004 | 24 | 0 |  |
| digitalservicebund/ris-search | 0011 | 15 | 0 |  |
| digitalservicebund/ris-search | 0014 | 15 | 0 |  |
| digitalservicebund/ris-search | 0017 | 11 | 0 |  |
| digitalservicebund/ris-search | 0020 | 10 | 0 |  |
| digitalservicebund/ris-search | 0021 | 14 | 0 |  |
| digitalservicebund/ris-search | 0022 | 13 | 0 |  |
| digitalservicebund/ris-search | 0024 | 5 | 0 |  |
| supabase/cli | 0013-live | 42 | 0 |  |
| supabase/cli | 0014 | 42 | 0 |  |
| supabase/cli | 0034 | 14 | 0 |  |
| supabase/cli | 0030 | 25 | 0 |  |
| FIWARE/data-space-connector | 0006 | 34 | 0 |  |
| FIWARE/data-space-connector | 0007 | 32 | 1 | 0007 |
| phel-lang/phel-lang | 0002 | 22 | 0 |  |
| phel-lang/phel-lang | 0003 | 25 | 0 |  |
| phel-lang/phel-lang | 0011 | 35 | 0 |  |
| phel-lang/phel-lang | 0012 | 26 | 0 |  |
