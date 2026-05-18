# Catalog Module

Model listing, metadata enrichment, and snapshot refresh for application-facing model catalogs.

## What This Module Does

The catalog layer separates two concerns:

1. **Native availability**
   Fetch the provider's own list-models endpoint to learn which raw model IDs are currently usable.

2. **Display metadata**
   Enrich those raw model IDs from a local snapshot built from a stable upstream catalog such as OpenRouter.

Applications should display the enriched data, but actual inference requests must still use the provider's original `raw_id`.

## Public Entry Points

- `list_models(request)`
  Fetch native provider models without enrichment.

- `list_models_with_enrichment(request, store)`
  Fetch native provider models and merge them with metadata from a `CatalogMetadataStore`.

- `enrich_manual_model(provider, raw_id, display_name, store)`
  Enrich a manually entered model ID from a `CatalogMetadataStore` without
  calling the provider's list-models endpoint first.

- `FileCatalogMetadataStore::load(path)`
  Load a local `catalog.json` snapshot into a file-backed metadata store.

- `refresh_catalog_snapshot(path, &CatalogRemoteConfig::default())`
  Refresh a local snapshot from the published remote manifest and snapshot files.

- `CatalogRemoteConfig::default()`
  Uses the default GitHub Pages manifest URL for this project.

## Recommended Application Flow

Use a stale-while-revalidate startup strategy:

1. Decide your own cache path for `catalog.json`.
2. Load the local snapshot if it exists.
3. Serve requests immediately using the local snapshot.
4. In the background, call `refresh_catalog_snapshot(...)`.
5. If the refresh returns `Updated`, reload the file-backed store.

`tiycore` does not guess your cache directory. The application should choose the local path and pass it in.

For ZenMux specifically, the library merges both the OpenAI-compatible
`/api/v1/models` list and the Vertex-style `/api/vertex-ai/v1beta/models` list
when fetching provider-native availability. This helps surface models such as
image-generation entries that may only appear in the Vertex-style list.

## Snapshot Files

The remote catalog publish flow uses two files:

- `manifest.json`
- `catalog.json`

The local application cache uses:

- your chosen `catalog.json` path
- an automatically derived local sidecar manifest path, such as `catalog.manifest.json`

## GitHub Pages Publishing

The default remote configuration assumes GitHub Pages serves:

- `catalog/manifest.json`
- `catalog/catalog.json`

This makes it easy for applications to use `CatalogRemoteConfig::default()` without hard-coding URLs.

## Snapshot Builder Tool

This crate also ships a small binary for catalog generation:

```bash
cargo run --bin tiy-catalog-sync -- \
  --output dist/catalog/catalog.json \
  --manifest-output dist/catalog/manifest.json \
  --snapshot-url catalog.json
```

Current behavior:

- Fetches OpenRouter's chat/completions model catalog and embeddings catalog
- Normalizes records into `CatalogModelMetadata`
- Writes a snapshot file and manifest file

This is intended for CI usage, such as a scheduled GitHub Actions workflow that publishes both files to GitHub Pages.

The repository workflow for this is:

- `.github/workflows/catalog-pages.yml`

## Example

```rust,no_run
use std::path::PathBuf;
use tiycore::catalog::{
    list_models_with_enrichment, load_catalog_metadata_store,
    refresh_catalog_snapshot, CatalogRemoteConfig, FetchModelsRequest,
};
use tiycore::Provider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot_path = PathBuf::from("/path/to/cache/catalog.json");

    let store = load_catalog_metadata_store(&snapshot_path)?;

    if let Some(store) = store.as_ref() {
        let result = list_models_with_enrichment(
            FetchModelsRequest::new(Provider::OpenAI),
            store,
        )
        .await?;
        println!("loaded {} models", result.models.len());
    }

    let _ = refresh_catalog_snapshot(&snapshot_path, &CatalogRemoteConfig::default()).await?;
    Ok(())
}
```

## Manual Model ID Enrichment

Applications sometimes need metadata enrichment even when no upstream
list-models API is available, or when a user types a model ID directly into the
UI.

In that case, skip `list_models_with_enrichment(...)` and call
`enrich_manual_model(...)` against the same local snapshot store:

```rust,no_run
use std::path::PathBuf;
use tiycore::catalog::{enrich_manual_model, load_catalog_metadata_store};
use tiycore::Provider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot_path = PathBuf::from("/path/to/cache/catalog.json");
    let store = load_catalog_metadata_store(&snapshot_path)?;

    if let Some(store) = store.as_ref() {
        let model = enrich_manual_model(
            Provider::OpenAI,
            "openai/gpt-4.1",
            None,
            store,
        );

        println!("raw_id = {}", model.raw_id);
        println!("display_name = {:?}", model.display_name);
        println!("context_window = {:?}", model.context_window);
    }

    Ok(())
}
```

Recommended application behavior:

1. Use enriched fields for display if a snapshot match is found.
2. Continue to persist and send the user-entered `raw_id` for inference calls.
3. If no snapshot match is found, treat the model as usable but partially
   enriched.

When the snapshot has been refreshed from OpenRouter, this same flow also works
for embedding model IDs such as `openai/text-embedding-3-small`.

## Smoke Check

You can verify the published GitHub Pages snapshot manually before wiring it
into an application:

```bash
curl -fsSL https://tiylabs.github.io/tiycore/catalog/manifest.json
curl -fsSL https://tiylabs.github.io/tiycore/catalog/catalog.json -o /tmp/catalog.json
python3 -m json.tool /tmp/catalog.json >/dev/null
```

For an application startup flow, the minimal stale-while-revalidate pattern
looks like this:

```rust,no_run
use std::path::PathBuf;
use tiycore::catalog::{
    load_catalog_metadata_store, refresh_catalog_snapshot, CatalogRemoteConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot_path = PathBuf::from("/path/to/cache/catalog.json");

    // 1. Load local snapshot if it already exists.
    let _local_store = load_catalog_metadata_store(&snapshot_path)?;

    // 2. Refresh in the background or during startup.
    let _refresh = refresh_catalog_snapshot(
        &snapshot_path,
        &CatalogRemoteConfig::default(),
    )
    .await?;

    // 3. Reload after a successful refresh if you need the newest snapshot now.
    let _updated_store = load_catalog_metadata_store(&snapshot_path)?;
    Ok(())
}
```
