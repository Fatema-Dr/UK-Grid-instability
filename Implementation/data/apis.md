
## ✅ **1. NESO 1-second Frequency Data – Where & How to Access**

### 📌 Source

The National Energy System Operator (NESO) publishes **historic system frequency data at 1-second resolution** on its Data Portal. ([National Energy System Operator (NESO)]

### 📍 API Endpoint Structure

NESO uses a **CKAN-based API** at:

```
https://api.neso.energy/api/3/action/
```

Useful endpoints (no authentication needed for open datasets): ([National Energy System Operator (NESO)][2])

| Purpose                     | API Path                                      |
| --------------------------- | --------------------------------------------- |
| List all datasets           | `/package_list`                               |
| Get details about a dataset | `/datapackage_show?id=<dataset_id>`           |
| Query data from a resource  | `/datastore_search?resource_id=<resource_id>` |
| SQL-style query             | `/datastore_search_sql?sql=SELECT ...`        |

❗ The **1-second frequency data isn’t a simple “single URL”**; it’s multiple resources you fetch via the CKAN API.

**Example placeholder URL to fetch data:**

```txt
https://api.neso.energy/api/3/action/datastore_search?
    resource_id=RESOURCE_ID_HERE
    &limit=10000
```

Replace `RESOURCE_ID_HERE` with the actual UUID for the given month/year. ([National Energy System Operator (NESO)][1])

---

## ✅ **2. UK Power Grid Inertia Cost Data – Where & How to Access**

### 📌 Source

NESO also publishes **system inertia cost data** — average daily price in £ per GVA.s — as CSV files on the Data Portal. ([National Energy System Operator (NESO)][3])

### 📍 API Access

Similarly to the frequency data, the inertia cost files are available via the **CKAN API**.

**Example CKAN query to fetch inertia costs:**

```txt
https://api.neso.energy/api/3/action/datastore_search?
    resource_id=INERTIA_COST_RESOURCE_ID
    &limit=10000
```

Where `INERTIA_COST_RESOURCE_ID` is the **UUID for a specific year’s inertia cost dataset**. ([National Energy System Operator (NESO)][4])

---

## ✅ **3. Required API Keys / Authentication**

### 🔑 NESO Data Portal

The **public data portal APIs (CKAN routes)** do *not require an API Key* for open datasets, including frequency and inertia cost, but they do have rate limits: ([National Energy System Operator (NESO)][2])

* **CKAN API**: ~1 request per second recommended
* **Datastore queries**: ~2 requests per minute recommended

⚠️ You only need authentication for paid/extranet APIs, which **you do not need for open historical data**. ([National Energy System Operator (NESO)][5])

**Example:**
If a particular dataset *did* require a key, an `Authorization:` HTTP header might look like:

```
Authorization: Token YOUR_API_KEY_HERE
```

(Replace `YOUR_API_KEY_HERE` with your actual key if needed — but not needed for the NESO open datasets.)

---

## ✅ **4. Response Format Expectations**

### 📄 CKAN API returns **JSON** by default

A typical JSON response from:

```
/datastore_search
```

looks like:

```json
{
  "help": "...",
  "success": true,
  "result": {
    "resource_id": "abc-123",
    "fields": [ { "id": "timestamp" }, { "id": "frequency" } ],
    "records": [
      { "timestamp": "2025-01-01T00:00:01Z", "frequency": 50.01 },
      ...
    ]
  }
}
```

The fields vary by dataset (e.g., frequency vs inertia cost). ([National Energy System Operator (NESO)][2])

CSV downloads are also available and may be simpler for bulk ingestion.

---

## ✅ **5. How to Specify Date Ranges / Filters**

### 📌 CKAN `datastore_search` filters

You can pass parameters such as:

```txt
limit=1000
filters={"DATE_COLUMN":"2025-01-01"}
```

Or via SQL:

```sql
SELECT * FROM "<resource_id>"
WHERE "timestamp" BETWEEN '2025-01-01' AND '2025-01-02'
```

Encoded in URL:

```
?sql=SELECT * FROM "<resource_id>" WHERE "timestamp" BETWEEN '2025-01-01' AND '2025-01-02'
```

These queries give you control over date ranges and other parameters. [National Energy System Operator (NESO)]

---

