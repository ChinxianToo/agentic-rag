## Data card: Annual CPI by division (2-digit), Malaysia

| Field | Value |
|--------|--------|
| **Dataset title** | Consumer Price Index (CPI) — annual, by division (2-digit COICOP-style groups) |
| **Description** | Long-format CSV: one row per (`date`, `division`) with CPI **index** value. Divisions include **`overall`** (headline) and numeric codes **`01`–`13`**. Years span **1960–2025** in the ingested extract (verify latest year against the portal when you refresh). |
| **Source URL(s)** | Primary catalogue: [https://open.dosm.gov.my/data-catalogue/cpi_annual](https://open.dosm.gov.my/data-catalogue/cpi_annual) · Portal home: [https://open.dosm.gov.my/](https://open.dosm.gov.my/) |
| **Local file ingested** | `project/parent_store/CPI 2D Annual.csv` (must match or be derived from the official download/API for this catalogue) |
| **License** | OpenDOSM typically publishes under **Creative Commons Attribution 4.0 International (CC BY 4.0)** — **confirm on the dataset page** at submission time and cite the exact license text linked there. |
| **Attribution** | Department of Statistics Malaysia (DOSM) via OpenDOSM; retain source URL and publication name in citations. |
| **Refresh cadence** | **Annual series**: updated when **annual** CPI figures are released and published on OpenDOSM (often aligned with the **annual CPI publication** cycle; not every monthly CPI release may change this extract). **Re-check** the catalogue “Last updated” field after each DOSM release. |
| **Last updated (portal)** | *Fill from OpenDOSM → `cpi_annual` → “Last updated” on submission date (e.g. web search snippets have mentioned **Jan 2025** for this catalogue — replace with the live value).* |
| **Last updated (this copy)** | *Set to the date you downloaded or exported `CPI 2D Annual.csv`.* |
| **Coverage** | Malaysia, national CPI index by division; **not** state-level unless you ingest a separate dataset. |
| **Known limitations** | Index level and base year must match DOSM documentation — do not assume base year without checking the official definition; early and recent years may use consistent chaining per DOSM methodology. |