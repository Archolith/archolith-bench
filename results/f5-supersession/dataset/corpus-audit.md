# F5 corpus: stripped spans for hand review

## FIWARE/data-space-connector 0001: 0001 - Authenticate participants to the consent-manager over OID4VP (2026-08-26, rule date_field)
path: `doc/adr/0001-oid4vp-for-participant-authentication.md`  status: ['**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Accepted)']
- L3 [status_line] `**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Accepted)`

## FIWARE/data-space-connector 0002: 0002 - Reuse the consent-facade's OID4VP client instead of implementing or importing OID4VP in Go (2026-08-26, rule date_field)
path: `doc/adr/0002-reuse-facade-oid4vp-client.md`  status: ['**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Accepted)']
- L3 [status_line] `**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Accepted)`

## FIWARE/data-space-connector 0003: 0003 - The facade exposes an internal token endpoint, not a consent proxy (2026-08-26, rule date_field)
path: `doc/adr/0003-token-endpoint-not-consent-proxy.md`  status: ['**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Accepted)']
- L3 [status_line] `**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Accepted)`

## FIWARE/data-space-connector 0004: 0004 - Map OID4VP tokens to participants by DID (2026-08-26, rule date_field)
path: `doc/adr/0004-map-oid4vp-tokens-to-participants-by-did.md`  status: ['**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Accepted (implemented))']
- L3 [status_line] `**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Accepted (implemented))`

## FIWARE/data-space-connector 0005: 0005 - Onboard participants with a verifiable credential, not client credentials (2026-08-26, rule date_field)
path: `doc/adr/0005-vc-based-participant-onboarding.md`  status: ['**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Proposed (design only - not implemented))']
- L3 [status_line] `**Status:** Superseded by [0008](0008-dsc-consent-manager.md) (was: Proposed (design only - not implemented))`

## FIWARE/data-space-connector 0006: 0006 - Validate eIDAS certificate chains inside VCVerifier, not in an external DSS service (2026-09-28, rule date_field)
path: `doc/adr/0006-eidas-validation-in-vcverifier.md`  status: ['**Status:** Accepted']
- (nothing removed)

## FIWARE/data-space-connector 0007: 0007 - Test the charts both rendered and applied, and installed as Helm releases (2026-10-14, rule date_field)
path: `doc/adr/0007-test-both-deployment-paths.md`  status: ['**Status:** Accepted']
- (nothing removed)

## FIWARE/data-space-connector 0008: 0008 - Replace the Node consent-manager with dsc-consent-manager on PostgreSQL (2026-11-02, rule date_field)
path: `doc/adr/0008-dsc-consent-manager.md`  status: ['**Status:** Accepted']
- L6 [meta_line] `**Supersedes:** [0001](0001-oid4vp-for-participant-authentication.md), [0002](0002-reuse-facade-oid4vp-client.md), [0003](0003-token-endpoint-not-consent-proxy.md), [0004](0004-map-oid4vp-tokens-to-participants-by-did.md), [0005](0005-vc-based-participant-onboarding.md)`

## digitalservicebund/digitalcheck-dito 0001: 1. Record architecture decisions (2024-05-16, rule first_iso_date_first_40_lines)
path: `doc/adr/0001-record-architecture-decisions.md`  status: ['- 2024-05-16: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0002: 2. Use React as the frontend framework (2024-05-16, rule first_iso_date_first_40_lines)
path: `doc/adr/0002-use-react.md`  status: ['- 2024-05-16: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0003: 3. Use Remix as the meta framework for React (2024-05-17, rule first_iso_date_first_40_lines)
path: `doc/adr/0003-use-remix.md`  status: ['- 2024-05-17: Accepted', '- 2025-03-20: Updated (see final section)', '- 2026-06-19: Superseded by [ADR 27](./0027-migrate-to-astro.md)']
- L7 [status_line] `- 2026-06-19: Superseded by [ADR 27](./0027-migrate-to-astro.md)`

## digitalservicebund/digitalcheck-dito 0004: 4. Use TypeScript (2024-05-18, rule first_iso_date_first_40_lines)
path: `doc/adr/0004-use-typescript.md`  status: ['- 2024-05-18: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0005: 5. Do not use an external dependency as state machine (2024-05-21, rule first_iso_date_first_40_lines)
path: `doc/adr/0005-no-state-machine-dependency.md`  status: ['- 2024-05-21: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0006: 6. Use separate routes for each page of the dialog form (2024-05-22, rule first_iso_date_first_40_lines)
path: `doc/adr/0006-use-separate-routes-for-dialog-form.md`  status: ['- 2024-05-22: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0007: 7. Store content in the repository without using a CMS (2024-05-23, rule first_iso_date_first_40_lines)
path: `doc/adr/0007-store-content-in-repository-without-cms.md`  status: ['- 2024-05-23: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0008: 8. Store content in a global file (2024-05-23, rule first_iso_date_first_40_lines)
path: `doc/adr/0008-store-content-in-global-file.md`  status: ['- 2024-05-23: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0009: 9. Use a Monorepo for All Digitalcheck Applications (2024-05-24, rule first_iso_date_first_40_lines)
path: `doc/adr/0009-use-monorepo.md`  status: ['- 2024-05-24: Accepted', '- 2025-03-03: Superceded by [0018](0018-switch-back-to-polyrepo.md)']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0010: 10. Make pipeline run dynamically for applications based on changes (2024-05-24, rule first_iso_date_first_40_lines)
path: `doc/adr/0010-make-pipeline-dynamic.md`  status: ['- 2024-05-24: Accepted', '- 2025-03-03: Superceded by [0018](0018-switch-back-to-polyrepo.md)']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0011: 11. Use Feature Flags (2024-06-22, rule first_iso_date_first_40_lines)
path: `doc/adr/0011-use-feature-flags.md`  status: ['- 2024-06-22: Accepted', '- 2025-11-24: Superseded by [ADR 25](./0025-use-feature-flags-k8s.md).']
- L6 [status_line] `- 2025-11-24: Superseded by [ADR 25](./0025-use-feature-flags-k8s.md).`

## digitalservicebund/digitalcheck-dito 0012: 12. Guidelines for testing (2024-06-22, rule first_iso_date_first_40_lines)
path: `doc/adr/0012-guidelines-for-testing.md`  status: ['- 2024-06-22: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0013: 13. Use RVF for implementing forms in React and Remix (2024-08-30, rule first_iso_date_first_40_lines)
path: `doc/adr/0013-use-RVF-for-forms.md`  status: ['- 2024-08-30: Accepted', '- 2025-03-20: Updated (see final section)']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0014: 14. Use of encryption (2024-09-06, rule first_iso_date_first_40_lines)
path: `doc/adr/0014-use-of-encryption.md`  status: ['- 2024-09-06: Accepted', '- 2025-12-11: Obsolete']
- L6 [status_line] `- 2025-12-11: Obsolete`

## digitalservicebund/digitalcheck-dito 0015: 15. Use a CMS for the new example pages content (2024-11-20, rule first_iso_date_first_40_lines)
path: `doc/adr/0015-use-cms-for-examples.md`  status: ['- 2024-11-20: Accepted', '- 2026-10-01: Superseded by [ADR 27](./0027-migrate-to-astro.md)']
- L6 [status_line] `- 2026-10-01: Superseded by [ADR 27](./0027-migrate-to-astro.md)`

## digitalservicebund/digitalcheck-dito 0016: 16. Use GraphQL Instead of REST for Data Requests to Strapi (2024-11-26, rule first_iso_date_first_40_lines)
path: `doc/adr/0016-use-graphql-for-strapi-requests.md`  status: ['- 2024-11-26: Accepted', '- 2026-10-01: Superseded by [ADR 27](./0027-migrate-to-astro.md)']
- L6 [status_line] `- 2026-10-01: Superseded by [ADR 27](./0027-migrate-to-astro.md)`

## digitalservicebund/digitalcheck-dito 0017: 17. Use Server-Side Caching with NodeCache for Strapi Data Fetching (2024-12-05, rule first_iso_date_first_40_lines)
path: `doc/adr/0017-use-serverside-caching.md`  status: ['- 2024-12-05: Accepted', '- 2026-06-19: Superseded by [ADR 27](./0027-migrate-to-astro.md)']
- L6 [status_line] `- 2026-06-19: Superseded by [ADR 27](./0027-migrate-to-astro.md)`

## digitalservicebund/digitalcheck-dito 0018: 18. Switch back from monorepo to polyrepo (2025-02-27, rule first_iso_date_first_40_lines)
path: `doc/adr/0018-switch-back-to-polyrepo.md`  status: ['- 2025-02-27: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0019: 19. Run e2e tests in parallel (2025-03-25, rule first_iso_date_first_40_lines)
path: `doc/adr/0019-run-e2e-tests-in-parallel.md`  status: ['- 2025-03-25: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0020: 20. Use more unit and snapshot tests instead of e2e-tests. (2025-05-20, rule first_iso_date_first_40_lines)
path: `doc/adr/0020-use-more-unit-and-snapshot-tests-instead-of-e2e.md`  status: ['- 2025-05-20: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0021: 21. Use custom goals for tracking (2025-06-13, rule first_iso_date_first_40_lines)
path: `doc/adr/0021-use-custom-goals-for-tracking.md`  status: ['- 2025-06-13: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0022: 22. Change routing and breadcrumb handling (2025-06-06, rule first_iso_date_first_40_lines)
path: `doc/adr/0022-change-routing-and-breadcrumbs.md`  status: ['- 2025-06-06: Accepted', '- 2026-06-19: Superseded by [ADR 27](./0027-migrate-to-astro.md)']
- L6 [status_line] `- 2026-06-19: Superseded by [ADR 27](./0027-migrate-to-astro.md)`

## digitalservicebund/digitalcheck-dito 0023: 23. Children-First React Component API (with Components-in-Props) (2025-08-19, rule first_iso_date_first_40_lines)
path: `doc/adr/0023-children-first-react-component-api.md`  status: ['- 2025-08-19: Accepted', '- 2025-12-08: Amended: use exported components, e.g., `Tab` instead of `Tabs.Tab`', '- 2026-03-18: Amended: update tab examples to the current `Tabs` API']
- L6 [status_clause] `- 2025-12-08: Amended: use exported components, e.g., `Tab` instead of `Tabs.Tab``
- L7 [status_continuation] `- 2026-03-18: Amended: update tab examples to the current `Tabs` API`

## digitalservicebund/digitalcheck-dito 0024: 24. Use localStorage to store data on the client (2025-10-08, rule first_iso_date_first_40_lines)
path: `doc/adr/0024-use-local-storage-for-data-storage.md`  status: ['- 2025-10-08: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0025: 25. Use Kubernetes ConfigMaps to set feature flags (2025-11-24, rule first_iso_date_first_40_lines)
path: `doc/adr/0025-use-feature-flags-k8s.md`  status: ['- 2025-11-24: Accepted', '- 2026-06-19: Superseded by [ADR 28](./0028-stage-based-feature-flags.md)']
- L6 [status_line] `- 2026-06-19: Superseded by [ADR 28](./0028-stage-based-feature-flags.md)`

## digitalservicebund/digitalcheck-dito 0026: 26. Use Headless UI for accessible UI primitives (2026-04-21, rule first_iso_date_first_40_lines)
path: `doc/adr/0026-use-headless-ui-for-accessible-ui-primitives.md`  status: ['- 2026-04-21: Accepted']
- (nothing removed)

## digitalservicebund/digitalcheck-dito 0027: 27. Migrate from React Router to Astro (2026-06-19, rule first_iso_date_first_40_lines)
path: `doc/adr/0027-migrate-to-astro.md`  status: ['- 2026-06-19: Accepted', '- 2026-10-01: Amended: replace Strapi with Astro content collections', '- Supersedes [ADR 3](./0003-use-remix.md) (meta framework), [ADR 15](./0015-use-cms-for-examples.md) (CMS), [ADR 16](./0016-use-graphql-for-strapi-requests.md) (GraphQL for Strapi), [ADR 17](./0017-use-serverside-caching.md) (server-side caching), [ADR 22](./0022-change-routing-and-breadcrumbs.md) (routing)']
- L6 [status_line] `- 2026-10-01: Amended: replace Strapi with Astro content collections`
- L7 [status_continuation] `- Supersedes [ADR 3](./0003-use-remix.md) (meta framework), [ADR 15](./0015-use-cms-for-examples.md) (CMS), [ADR 16](./0016-use-graphql-for-strapi-requests.md) (GraphQL for Strapi), [ADR 17](./0017-use-serverside-caching.md) (server-side caching), [ADR 22](./0022-change-routing-and-breadcrumbs.md) (routing)`

## digitalservicebund/digitalcheck-dito 0028: 28. Replace feature flags with staging-only features (2026-06-19, rule first_iso_date_first_40_lines)
path: `doc/adr/0028-stage-based-feature-flags.md`  status: ['- 2026-06-19: Accepted', '- Supersedes [ADR 25](./0025-use-feature-flags-k8s.md) (Kubernetes ConfigMap flags), which superseded [ADR 11](./0011-use-feature-flags.md) (Unleash)']
- L6 [meta_line] `- Supersedes [ADR 25](./0025-use-feature-flags-k8s.md) (Kubernetes ConfigMap flags), which superseded [ADR 11](./0011-use-feature-flags.md) (Unleash)`

## digitalservicebund/ris-search 0001: 1. Record architecture decisions (2021-10-06, rule date_field)
path: `doc/adr/0001-record-architecture-decisions.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0002: 2. Use Gradle as build tool (2021-10-06, rule date_field)
path: `doc/adr/0002-use-gradle-as-build-tool.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0003: 3. Use Java 17 (2022-05-03, rule date_field)
path: `doc/adr/0003-use-java-17.md`  status: ['Superseded by [ADR 12](./0012-use-java-21.md).']
- L7 [meta_line] `Superseded by [ADR 12](./0012-use-java-21.md).`

## digitalservicebund/ris-search 0004: 4. Dockerized the Backend and Frontend Development Environment (2024-07-23, rule date_field)
path: `doc/adr/0004-dockerizing-development-environment.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0005: 5. Migrated Dependencies to a TOML File (2024-08-07, rule date_field)
path: `doc/adr/0005-migrated-dependencies-to-toml-file.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0006: 6. Universal Search API Endpoint (2024-08-21, rule date_field)
path: `doc/adr/0006-universal-search-api-endpoint.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0007: 7. Using Pact Library for Contract Testing (2024-09-25, rule date_field)
path: `doc/adr/0007-using-pact-for-contract-testing.md`  status: ['Superseded by [ADR 19](./0019-remove-pact-contract-testing.md).']
- L7 [meta_line] `Superseded by [ADR 19](./0019-remove-pact-contract-testing.md).`

## digitalservicebund/ris-search 0008: 8. Use manifestations as the primary entity for legislation (2024-11-01, rule date_field)
path: `doc/adr/0008-use-manifestations-as-the-primary-entity-for-legislation.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0009: 9. Automatic class sorting with Prettier (2025-01-06, rule date_field)
path: `doc/adr/0009-automatic-class-sorting-with-prettier.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0010: 10. Consolidate Frontend Code into a Single Folder Structure (2024-12-05, rule date_field)
path: `doc/adr/0010-merge-shared-public-internal-folders-frontend.md`  status: ['Superseded by [ADR 16](./0016-use-feature-flags-instead-of-profiles.md) regarding the `NUXT_PUBLIC_PROFILE`', 'variable and its helpers; the single-folder frontend structure itself remains in place.']
- L7 [meta_line] `Superseded by [ADR 16](./0016-use-feature-flags-instead-of-profiles.md) regarding the `NUXT_PUBLIC_PROFILE``
- L8 [meta_continuation] `variable and its helpers; the single-folder frontend structure itself remains in place.`

## digitalservicebund/ris-search 0011: 11. Run end-to-end tests in pipeline (2024-12-19, rule date_field)
path: `doc/adr/0011-run-end-to-end-tests-in-pipeline.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0012: 12. Use Java 21 (2025-01-29, rule date_field)
path: `doc/adr/0012-use-java-21.md`  status: ['Superseded by [ADR 25](./0025-use-java-25.md).']
- L7 [meta_line] `Superseded by [ADR 25](./0025-use-java-25.md).`

## digitalservicebund/ris-search 0013: 13. S3 Buckets and OpenSearch Synchronization (2025-03-07, rule date_field)
path: `doc/adr/0013-s3-opensearch-Synchronization.md`  status: ['Superseded by ADR 0015']
- L7 [meta_line] `Superseded by ADR 0015`

## digitalservicebund/ris-search 0014: 14. Replace Talisman with GitHub Secrets Check (2025-05-07, rule date_field)
path: `doc/adr/0014-replace-talisman-with-github-secrets-check.md`  status: ['Decided']
- (nothing removed)

## digitalservicebund/ris-search 0015: 15. S3 Buckets and OpenSearch Synchronization (2025-05-05, rule date_field)
path: `doc/adr/0015-s3-opensearch-Synchronization.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0016: 16. Replace `NUXT_PUBLIC_PROFILE` with `PRIVATE_FEATURES_ENABLED` (2025-10-30, rule date_field)
path: `doc/adr/0016-use-feature-flags-instead-of-profiles.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0017: 17. Remove OAuth2 in Favor of Basic Auth for Staging (2025-11-04, rule date_field)
path: `doc/adr/0017-remove-oauth2-from-backend-and-frontend.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0018: 18. Remove Dependency of End-to-End Tests on Prebuilt Docker Images (2025-11-20, rule date_field)
path: `doc/adr/0018-remove-dependency-of-e2e-tests-on-prebuild-docker-images.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0019: 19. Removing Pact Contract Testing in Favor of Integration & E2E Tests (2025-12-05, rule date_field)
path: `doc/adr/0019-remove-pact-contract-testing.md`  status: ['**Status:** Accepted']
- L5 [meta_line] `**Supersedes:** ADR-007 “Using Pact Library for Contract Testing”`
- L86 [body_sentence] `ADR-007 is now deprecated.`

## digitalservicebund/ris-search 0020: 20. Migrate Frontend Package Manager from Yarn to PNPM (2026-01-20, rule date_field)
path: `doc/adr/0020-pnpm-migration.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0021: 21. Adopt Oxlint for frontend linting (2026-01-22, rule date_field)
path: `doc/adr/0021-adopt-oxlint-for-frontend-linting.md`  status: ['Accepted (experimental support added in [PR #1600](https://github.com/digitalservicebund/ris-search/pull/1600))']
- (nothing removed)

## digitalservicebund/ris-search 0022: 22. Generate TypeScript types from OpenAPI spec (2026-03-04, rule date_field)
path: `doc/adr/0022-generated-openapi-types.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0023: 23. Use NormTestDataBuilder for norm LDML test data (2026-07-13, rule date_field)
path: `doc/adr/0023-norm-test-data-builder.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0024: 24. Use German for domain-specific terms in code (2026-10-05, rule date_field)
path: `doc/adr/0024-use-german-for-domain-specific-terms-in-code.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0025: 25. Use Java 25 (2026-08-13, rule date_field)
path: `doc/adr/0025-use-java-25.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0026: 26. Frontend analytics with PostHog (2026-10-29, rule date_field)
path: `doc/adr/0026-frontend-analytics-with-posthog.md`  status: ['Accepted']
- (nothing removed)

## digitalservicebund/ris-search 0027: 27. HTML processing pipeline in the frontend (2026-11-17, rule date_field)
path: `doc/adr/0027-html-processing-pipeline.md`  status: ['Proposed']
- (nothing removed)

## digitalservicebund/ris-search 0027: 27. JSON-LD with German fields and schema.org mappings (2026-11-11, rule date_field)
path: `doc/adr/0027-json-ld-with-german-fields-and-schema-org.md`  status: ['Proposed']
- (nothing removed)

## phel-lang/phel-lang 0001: ADR 0001: Record architecture decisions (2026-07-29, rule date_field)
path: `docs/adr/0001-record-architecture-decisions.md`  status: ['- **Status**: Accepted']
- L22 [body_sentence] `ADRs are immutable and superseded rather than edited.`

## phel-lang/phel-lang 0002: ADR 0002: Compile to PHP source, not a bytecode VM (2026-07-29, rule date_field)
path: `docs/adr/0002-compile-to-php-source.md`  status: ['- **Status**: Accepted (recorded retroactively)']
- (nothing removed)

## phel-lang/phel-lang 0003: ADR 0003: Modules talk to each other through facades only (2026-07-29, rule date_field)
path: `docs/adr/0003-modules-talk-through-facades.md`  status: ['- **Status**: Accepted (recorded retroactively)']
- (nothing removed)

## phel-lang/phel-lang 0004: ADR 0004: Accept four module cycles and pin them (2026-07-29, rule date_field)
path: `docs/adr/0004-accept-four-module-cycles.md`  status: ['- **Status**: Accepted (recorded retroactively; decided in #2785 and earlier)']
- (nothing removed)

## phel-lang/phel-lang 0005: ADR 0005: Define the public PHP API by rule, gate it by snapshot (2026-07-29, rule date_field)
path: `docs/adr/0005-public-php-api-by-rule-and-snapshot.md`  status: ['- **Status**: Accepted (recorded retroactively; landed in the run-up to 1.0)']
- (nothing removed)

## phel-lang/phel-lang 0006: ADR 0006: One opt-in channel for compiler deprecations (2026-07-29, rule date_field)
path: `docs/adr/0006-one-opt-in-deprecation-channel.md`  status: ['- **Status**: Accepted, amended by [0014](0014-announce-the-separator-deprecation.md)']
- L3 [status_clause] `- **Status**: Accepted, amended by [0014](0014-announce-the-separator-deprecation.md)`
- L4 [status_continuation] `and [0017](0017-a-diagnostic-renders-itself-on-stderr.md)`
- L7 [meta_line] `- **Amended by**: [0014](0014-announce-the-separator-deprecation.md) — the `\``
- L8 [meta_continuation] `separator announces whether or not the flag is set, so "off by default" now has`
- L9 [meta_continuation] `exactly one exception. Everything else here holds, and 0014's first-party`
- L10 [meta_continuation] `scoping is the precondition this record named.`
- L11 [meta_line] `- **Amended by**: [0017](0017-a-diagnostic-renders-itself-on-stderr.md) —`
- L12 [meta_continuation] ``ErrorNotice::raise()` writes the notice to stderr itself instead of letting`
- L13 [meta_continuation] `PHP render a `trigger_error()`, so decision point 2 below names a mechanism`
- L14 [meta_continuation] `that only a userland error handler still reaches.`
- SURVIVES stripping (contains 'supersed'):
  - `- `tests/php/Unit/Compiler/Deprecation/`, including `SupersededFormDeprecatorTest``
  - `- `SupersededFormDeprecationTest`, `MacroExpansionDeprecationTest` (attribution)`
  - ``SupersededFormDeprecator``

## phel-lang/phel-lang 0007: ADR 0007: Clojure-style interop is the source spelling (2026-07-29, rule date_field)
path: `docs/adr/0007-clojure-style-interop-is-the-source-spelling.md`  status: ['- **Status**: Accepted (recorded retroactively; decided across #2875 to #2887)']
- (nothing removed)
- SURVIVES stripping (contains 'supersed'):
  - `emitted on your behalf a moment later. `SupersededFormDeprecator` therefore runs`
  - ``SupersededFormDeprecator`, and fails on a spec row with no dispatch entry, which`
  - `- `SupersededFormDeprecationTest`, `QualifiedMemberValueRuntimeTest``

## phel-lang/phel-lang 0008: ADR 0008: The namespace separator is `.`, and `\` still parses (2026-07-29, rule date_field)
path: `docs/adr/0008-dot-namespace-separator.md`  status: ['- **Status**: Accepted, amended by [0014](0014-announce-the-separator-deprecation.md)']
- L3 [status_clause] `- **Status**: Accepted, amended by [0014](0014-announce-the-separator-deprecation.md)`
- L6 [meta_line] `- **Amended by**: [0014](0014-announce-the-separator-deprecation.md) — the warning`
- L7 [meta_continuation] `is no longer opt-in, and [0015](0015-a-php-class-is-named-with-dots.md) fixes the`
- L8 [meta_continuation] `removal to the next major rather than to a minor boundary.`

## phel-lang/phel-lang 0009: ADR 0009: Write the standard library in Phel, ship it precompiled (2026-07-29, rule date_field)
path: `docs/adr/0009-stdlib-in-phel-precompiled-in-the-phar.md`  status: ['- **Status**: Accepted (recorded retroactively; precompilation landed in #2443)']
- (nothing removed)

## phel-lang/phel-lang 0010: ADR 0010: Static analysis covers `src/` only (2026-07-29, rule date_field)
path: `docs/adr/0010-static-analysis-covers-src-only.md`  status: ['- **Status**: Accepted (recorded retroactively)']
- (nothing removed)

## phel-lang/phel-lang 0011: ADR 0011: Persistent collections implemented in PHP (2026-07-29, rule date_field)
path: `docs/adr/0011-persistent-collections-in-php.md`  status: ['- **Status**: Accepted (recorded retroactively)']
- (nothing removed)

## phel-lang/phel-lang 0012: ADR 0012: Non-hygienic macros with auto-gensym (2026-07-29, rule date_field)
path: `docs/adr/0012-non-hygienic-macros-with-auto-gensym.md`  status: ['- **Status**: Accepted (recorded retroactively)']
- (nothing removed)

## phel-lang/phel-lang 0013: ADR 0013: A static property is `$prop` to read and a plain name to assign (2026-08-01, rule date_field)
path: `docs/adr/0013-static-property-spelling.md`  status: ['- **Status**: Accepted']
- L5 [meta_line] `- **Amends**: [0007](0007-clojure-style-interop-is-the-source-spelling.md) — closes`
- L6 [meta_continuation] `the assignable-static-property question that record left open.`

## phel-lang/phel-lang 0014: ADR 0014: The `\` separator deprecation announces by default (2026-08-02, rule date_field)
path: `docs/adr/0014-announce-the-separator-deprecation.md`  status: ['- **Status**: Accepted']
- L5 [meta_line] `- **Amends**: [0006](0006-one-opt-in-deprecation-channel.md) — one exception to`
- L6 [meta_continuation] `the opt-in rule, on the conditions below. The rest of 0006 stands.`

## phel-lang/phel-lang 0015: ADR 0015: A PHP class is named with dots and no leading marker (2026-08-02, rule date_field)
path: `docs/adr/0015-a-php-class-is-named-with-dots.md`  status: ['- **Status**: Accepted. Reviewed by @jasalt, including the lower-case vendor']
- L7 [meta_line] `- **Supersedes**: none. Settles the question [ADR 0008](0008-dot-namespace-separator.md) left open.`

## phel-lang/phel-lang 0016: ADR 0016: A bare all-caps host name reads by position, not by probing (2026-08-16, rule date_field)
path: `docs/adr/0016-a-bare-all-caps-host-name-reads-by-position.md`  status: ['- **Status**: Accepted.']
- (nothing removed)

## phel-lang/phel-lang 0017: ADR 0017: A compiler diagnostic renders itself on stderr (2026-09-09, rule date_field)
path: `docs/adr/0017-a-diagnostic-renders-itself-on-stderr.md`  status: ['- **Status**: Accepted']
- L5 [meta_line] `- **Amends**: [0006](0006-one-opt-in-deprecation-channel.md) — its decision point`
- L6 [meta_continuation] `2 named `trigger_error()` as the rendering mechanism. One channel, one owner,`
- L7 [meta_continuation] `and the `display_errors` guard all stand; only who writes the line changes.`

## phel-lang/phel-lang 0018: ADR 0018: A superseded form is rejected as source and kept as the target (2026-09-11, rule date_field)
path: `docs/adr/0018-superseded-forms-are-rejected-as-source.md`  status: ['- **Status**: Accepted']
- L5 [meta_line] `- **Amends**: [ADR 0007](0007-clojure-style-interop-is-the-source-spelling.md)`
- L9 [body_sentence] `ADR 0007 made the Clojure-style spellings the only ones to write and deprecated`
- L10 [sentence_continuation] ``php/new`, `php/->` and `php/::` as source.`
- SURVIVES stripping (contains 'supersed'):
  - `# ADR 0018: A superseded form is rejected as source and kept as the target`
  - `- `SupersededFormRejectorTest` covers the walk: each of the four heads, a head`
  - `- `SupersededFormRejectionTest` compiles every Clojure-style shorthand and both`
  - ``SupersededFormRejector::supersededFormNames()`.`

## phel-lang/phel-lang 0019: ADR 0019: `withMeta` returns a copy (2026-09-21, rule date_field)
path: `docs/adr/0019-with-meta-returns-a-copy.md`  status: ['- **Status**: Accepted']
- (nothing removed)

## phel-lang/phel-lang 0020: ADR 0020: Build artifacts require the generating Phel version (2026-10-05, rule date_field)
path: `docs/adr/0020-build-artifacts-require-the-generating-version.md`  status: ['- **Status**: Accepted']
- L5 [meta_line] `- **Amends**: 0005 (compatibility of generated artifacts)`

## phel-lang/phel-lang 0021: ADR 0021: Expose embedding types and exclude internal tooling (2026-10-05, rule date_field)
path: `docs/adr/0021-public-embedding-types-and-internal-tooling.md`  status: ['- **Status**: Accepted']
- L5 [meta_line] `- **Supersedes**: [ADR 0005](0005-public-php-api-by-rule-and-snapshot.md)`
- SURVIVES stripping (contains 'supersed'):
  - `Shared Performance classes, SupersededSourceMaps and LintRuleExplainerInterface`

## phel-lang/phel-lang 0022: ADR 0022: The CLI's machine surface is under semver, its human text is not (2026-10-08, rule date_field)
path: `docs/adr/0022-the-cli-machine-surface-is-under-semver.md`  status: ['- **Status**: Accepted']
- (nothing removed)

## phel-lang/phel-lang 0023: ADR 0023: Build source paths follow the artifact (2026-11-14, rule date_field)
path: `docs/adr/0023-build-source-paths-follow-the-artifact.md`  status: ['- **Status**: Accepted']
- (nothing removed)

## supabase/cli 0000: 0000. Use ADR to Record Architectural Decisions (2026-02-10, rule date_field)
path: `docs/adr/0000-use-adr-to-record-decisions.md`  status: ['**Status**: accepted']
- (nothing removed)
- SURVIVES stripping (contains 'supersed'):
  - `- **Evolvable** — decisions can be marked as deprecated or superseded`

## supabase/cli 0001: 0001. CLI DX Architecture: The 7 Pillars (2026-02-10, rule date_field)
path: `docs/adr/0001-cli-dx-architecture-pillars.md`  status: ['**Status**: accepted']
- (nothing removed)

## supabase/cli 0002: 0002. CLI Product Metrics (2026-02-10, rule date_field)
path: `docs/adr/0002-cli-product-metrics.md`  status: ['**Status**: accepted']
- (nothing removed)

## supabase/cli 0003: 0003. Self-Documenting CLI & Documentation Strategy (2026-02-10, rule date_field)
path: `docs/adr/0003-self-documenting-cli.md`  status: ['**Status**: accepted']
- (nothing removed)

## supabase/cli 0004: 0004. CLI Design Goals & Development Workflows (2026-02-10, rule date_field)
path: `docs/adr/0004-cli-design-goals-and-workflows.md`  status: ['**Status**: accepted, amended by [0033](0033-target-selection-and-hosted-mutation-consent.md)']
- L3 [status_clause] `**Status**: accepted, amended by [0033](0033-target-selection-and-hosted-mutation-consent.md)`

## supabase/cli 0005: 0005. OpenAPI-Driven Code Generation for CRUD Commands (2026-02-10, rule date_field)
path: `docs/adr/0005-openapi-driven-code-generation.md`  status: ['**Status**: proposed']
- (nothing removed)

## supabase/cli 0006: 0006. Environment Management & Variable Resolution (2026-02-10, rule date_field)
path: `docs/adr/0006-environment-management.md`  status: ['**Status**: proposed']
- (nothing removed)

## supabase/cli 0007: 0007. Real-time Progress in Command Handlers (2026-02-10, rule date_field)
path: `docs/adr/0007-realtime-progress-in-command-handlers.md`  status: ['**Status**: proposed', 'status: "running" | "done" | "failed";']
- (nothing removed)

## supabase/cli 0008: 0008. Authentication & Token Management (2026-02-10, rule date_field)
path: `docs/adr/0008-authentication-and-token-management.md`  status: ['**Status**: proposed']
- (nothing removed)

## supabase/cli 0009: 0009. Configuration Schema & Validation (2026-02-10, rule date_field)
path: `docs/adr/0009-configuration-schema-and-validation.md`  status: ['**Status**: proposed']
- (nothing removed)

## supabase/cli 0011: 0011. CLI Release & Distribution Strategy (2026-04-17, rule date_field)
path: `docs/adr/0011-cli-release-and-distribution-strategy.md`  status: ['**Status**: accepted']
- L34 [body_sentence] `| Dist-tags                      | **Superseded by [ADR 0028](0028-release-branches-and-maintenance-lines.md).**`
- L36 [body_sentence] `| Artifact signing               | **Superseded by [ADR 0014](0014-macos-code-signing-and-notarization.md).**`
- L89 [meta_line] `> **Superseded by [ADR 0028](0028-release-branches-and-maintenance-lines.md)**, which adds the `next` and `v<N>.stable` dist-tags and retires the "next shell" `alpha` wording below.`

## supabase/cli 0013: 0013. Hybrid Stitch + Stamp for Telemetry Identity Attribution (2026-06-11, rule date_field)
path: `docs/adr/0013-hybrid-stitch-stamp-identity-attribution.md`  status: ['**Status**: proposed']
- (nothing removed)

## supabase/cli 0013: 0013. Live E2E Tests Bypass the Replay Server (2026-06-16, rule date_field)
path: `docs/adr/0013-live-e2e-bypasses-replay-server.md`  status: ['**Status**: accepted']
- (nothing removed)

## supabase/cli 0014: 0014. macOS Code Signing & Notarization (2026-06-24, rule date_field)
path: `docs/adr/0014-macos-code-signing-and-notarization.md`  status: ['**Status**: accepted']
- L35 [body_sentence] `This supersedes only the **"Artifact signing"** row and **contingent follow-up A** of ADR 0011.`
- L77 [body_sentence] `This ADR supersedes its "Artifact signing" decision and contingent follow-up A.`

## supabase/cli 0015: 0015. Managed Stack Contract Fixtures (2026-08-10, rule date_field)
path: `docs/adr/0015-managed-stack-contract-fixtures.md`  status: ['**Status**: superseded by [ADR-0017](0017-simplified-managed-stack-architecture.md)']
- L3 [status_line] `**Status**: superseded by [ADR-0017](0017-simplified-managed-stack-architecture.md)`

## supabase/cli 0016: 0016. Legacy Port Completion and Go CLI Authority Scope (2026-08-11, rule date_field)
path: `docs/adr/0016-legacy-port-completion-and-go-cli-authority-scope.md`  status: ['**Status**: accepted (2026-08-11), superseded by [0031](0031-go-cli-removal.md)']
- L3 [status_clause] `**Status**: accepted (2026-08-11), superseded by [0031](0031-go-cli-removal.md)`

## supabase/cli 0017: 0017. Simplified managed stack architecture (2026-08-17, rule date_field)
path: `docs/adr/0017-simplified-managed-stack-architecture.md`  status: ['**Status**: superseded by [the Stack package architecture](../../packages/stack/ARCHITECTURE.md)']
- L3 [status_line] `**Status**: superseded by [the Stack package architecture](../../packages/stack/ARCHITECTURE.md)`

## supabase/cli 0018: 0018. Sparse Config Subtraction (2026-08-18, rule date_field)
path: `docs/adr/0018-sparse-config-subtraction.md`  status: ['**Status**: proposed']
- L28 [body_sentence] `(This ADR originally assigned the translation of that shape to `ProjectConfig` to the diff core, CLI-2156; ADR 0019 supersedes that assignment — the mapping lives in `@supabase/config` as a separate layer so the CLI and Studio share one mapper.`
- L61 [body_sentence] `- [ADR 0019](0019-config-api-response-passthrough.md): Raw API-Response Passthrough — supersedes this ADR's assignment of API→`ProjectConfig` translation to the diff core (CLI-2156): the mapping lives in `@supabase/config` so the CLI and Studio share one implementation, while the subtraction core stays independent of the API shape`

## supabase/cli 0019: 0019. Raw API-response passthrough on API-sourced config (`_apiResponse`) (2026-08-24, rule date_field)
path: `docs/adr/0019-config-api-response-passthrough.md`  status: ['**Status**: accepted']
- L219 [body_sentence] `This ADR supersedes 0018's`
- L220 [sentence_continuation] `assignment of API→`ProjectConfig` translation to the diff core (CLI-2156):`
- L221 [sentence_continuation] `the mapping lives in `@supabase/config` so the CLI and Studio share one`
- L222 [sentence_continuation] `implementation.`

## supabase/cli 0020: 0020. Config naming vocabulary: `CliConfig`, `ProjectConfig`, `CliSettings`, and the `Cli*` prefix rule (2026-08-24, rule date_field)
path: `docs/adr/0020-config-naming-vocabulary.md`  status: ['**Status**: accepted']
- (nothing removed)

## supabase/cli 0021: 0021. `ProjectConfig` normalizers predict post-push state, not a verbatim projection (2026-08-27, rule date_field)
path: `docs/adr/0021-projectconfig-convergence-semantics.md`  status: ['**Status**: accepted']
- (nothing removed)
- SURVIVES stripping (contains 'supersed'):
  - `OMITTED unconditionally, regardless of `enabled` (`applyPushUnmanagedOmissions`, superseding the`

## supabase/cli 0022: 0022. Config Diff Classification and Managed Surface (2026-08-20, rule date_field)
path: `docs/adr/0022-config-diff-classification-and-managed-surface.md`  status: ['**Status**: accepted']
- (nothing removed)
- SURVIVES stripping (contains 'supersed'):
  - `3. **The original self-contained translation table** (this ADR's first accepted form): correct, but once CLI-2230 landed the registry it became a ~600-line parallel implementation of the same mapping with independently-drifting transforms. Superseded by the consolidation above.`

## supabase/cli 0023: 0023. `config pull` Write Strategy and Scope Resolution (2026-09-02, rule date_field)
path: `docs/adr/0023-config-pull-write-strategy-and-scope-resolution.md`  status: ['**Status**: accepted']
- (nothing removed)
- SURVIVES stripping (contains 'supersed'):
  - `**Addendum (2026-11-04): JSON and JSONC edits are span-edited too.** Targeted edits to `config.json` and `config.jsonc` now go through `jsonc-parser`'s `modify`/`applyEdits` using the file's own detected indent and newline, so comments, trailing commas, and untouched values keep their bytes. The for`

## supabase/cli 0024: 0024. Top-Level `pull` Orchestration (2026-09-08, rule date_field)
path: `docs/adr/0024-top-level-pull-orchestration.md`  status: ['**Status**: accepted']
- L57 [meta_line] `supersedes ADR 0004's "runs all sub-syncs in parallel" line for `pull` specifically, for three`
- SURVIVES stripping (contains 'supersed'):
  - ``config pull`'s own flag of the same name — superseding the original "Non-Goals" section's`
  - ``supabase/config.toml`** — superseding section 4's "`--force` (bypass the config git-dirty guard)"`

## supabase/cli 0025: 0025. Ephemeral Postgres for schema tooling (2026-09-09, rule date_field)
path: `docs/adr/0025-ephemeral-postgres-for-schema-tooling.md`  status: ['**Status**: superseded by throwaway stack shadow stacks and [database snapshots](../../packages/stack/ARCHITECTURE.md#snapshots-belong-to-the-database-instance) - `EphemeralPostgres` was never built; see [`stack-shadow.ts`](../../apps/cli/src/command-internal/stack-shadow.ts); runtime auto-selection and native-as-root handling amended by [ADR 0029](0029-native-postgres-privilege-step-down.md)']
- L3 [status_clause] `**Status**: superseded by throwaway stack shadow stacks and [database snapshots](../../packages/stack/ARCHITECTURE.md#snapshots-belong-to-the-database-instance) — `EphemeralPostgres` was never built; see [`stack-shadow.ts`](../../apps/cli/src/command-internal/stack-shadow.ts); runtime auto-selection and native-as-root handling amended by [ADR 0029](0029-native-postgres-privilege-step-down.md)`

## supabase/cli 0026: 0026. Slim image and native artifact mirrors (2026-09-29, rule date_field)
path: `docs/adr/0026-slim-artifact-mirrors.md`  status: ['**Status**: accepted']
- L5 [meta_line] `**Amended**: 2026-10-15 — stack mode resolves versions into a project lock file; see "The lock`
- SURVIVES stripping (contains 'supersed'):
  - `upgrade first and the hotfix PR is superseded and must be closed by hand.`
  - `recomputes from the latest develop: a stale plan can never be applied, and a superseded PR's branch`
  - `one — a superseded PR is never auto-closed. Commits someone else pushed to the open PR, such as a`
  - `superseded commits are dropped, and a replay conflict fails the run for a manual rebase. A backlog`

## supabase/cli 0027: 0027. CLI tracing conventions (2026-09-29, rule date_field)
path: `docs/adr/0027-cli-tracing-conventions.md`  status: ['**Status**: proposed']
- (nothing removed)

## supabase/cli 0028: 0028. Release branches and maintenance lines (2026-10-07, rule date_field)
path: `docs/adr/0028-release-branches-and-maintenance-lines.md`  status: ['**Status**: accepted']
- L28 [body_sentence] `This supersedes the dist-tag section of ADR 0011 ("Why `latest`, `beta`, and `alpha` npm dist-tags").`
- L66 [body_sentence] `- [ADR 0011](0011-cli-release-and-distribution-strategy.md): the release and distribution strategy; its dist-tag section is superseded here.`

## supabase/cli 0029: 0029. Native Postgres privilege step-down (2026-10-07, rule date_field)
path: `docs/adr/0029-native-postgres-privilege-step-down.md`  status: ['**Status**: proposed']
- (nothing removed)

## supabase/cli 0030: 0030. Hotfix releases (2026-10-07, rule date_field)
path: `docs/adr/0030-hotfix-releases.md`  status: ['**Status**: accepted']
- (nothing removed)

## supabase/cli 0031: 0031. Go CLI Removal (2026-09-14, rule date_field)
path: `docs/adr/0031-go-cli-removal.md`  status: ['**Status**: accepted']
- L122 [meta_line] `- Supersedes [ADR 0016](0016-legacy-port-completion-and-go-cli-authority-scope.md), whose`
- L123 [meta_continuation] `residual authority scope no longer applies now that `apps/cli-go/` is gone.`

## supabase/cli 0032: 0032. Config value precedence (2026-10-08, rule date_field)
path: `docs/adr/0032-config-value-precedence.md`  status: ['**Status**: proposed']
- (nothing removed)
- SURVIVES stripping (contains 'supersed'):
  - `The prompt before seeding into the matched project is superseded by the single hosted-mutation`
  - `and the hosted-mutation consent that supersedes the seed consent above.`

## supabase/cli 0033: 0033. Target selection and hosted-mutation consent (2026-10-15, rule date_field)
path: `docs/adr/0033-target-selection-and-hosted-mutation-consent.md`  status: ['**Status**: proposed']
- L292 [body_sentence] `- [ADR 0004](0004-cli-design-goals-and-workflows.md): the safety model this decision amends.`
- L293 [body_sentence] `This ADR supersedes its`
- L294 [sentence_continuation] `seed consent section.`

## supabase/cli 0034: 0034. Config diagnostics print no source text (2026-11-11, rule date_field)
path: `docs/adr/0034-config-diagnostics-print-no-source-text.md`  status: ['**Status**: accepted']
- (nothing removed)
