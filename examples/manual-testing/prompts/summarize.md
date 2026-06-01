Summarize the following technical discussion:

The team met to discuss the migration from PostgreSQL 14 to PostgreSQL 16.
Key points raised:

1. The pg_stat_statements extension needs to be updated first, as the current
   version is incompatible with PG16. DBA team estimates 2 hours of downtime
   for the extension upgrade alone.

2. Three stored procedures use deprecated syntax (the old-style implicit
   casting in CREATE CAST). These need to be rewritten before migration.
   Sarah volunteered to handle this by end of sprint.

3. The connection pooler (PgBouncer 1.18) has a known issue with PG16's
   new query protocol changes. We need to upgrade to PgBouncer 1.21+.
   This affects all 12 application services that connect through the pooler.

4. Read replicas are currently on PG14.9. The plan is to promote one replica,
   upgrade it, then add it back. Rolling upgrade across 4 replicas will take
   approximately 6 hours with the current automation scripts.

5. Application-level changes: the ORM layer uses a query pattern that triggers
   a planner regression in PG16.0 (fixed in 16.2). We must target 16.2 or
   later. CI tests should be run against PG16.2 before any production changes.

6. Rollback plan: snapshot all volumes before upgrade, keep PG14 binaries
   on the host for 72 hours. If rollback is needed, stop PG16, restore
   snapshot, start PG14. Estimated rollback time: 15 minutes.

7. Timeline: prep work in sprint 23, migration window requested for the
   Saturday of sprint 24 (March 15). All teams must sign off by March 10.
