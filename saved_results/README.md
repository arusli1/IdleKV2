# Saved Results

This directory holds small tracked copies of the canonical P0-P3 summary
artifacts.

It is not meant to replace the full generated `results/` trees under each
phase directory. It exists so the repo keeps a durable, lightweight memory of
what has already been run even if local generated outputs are later pruned.

It also keeps the small `phase3_pilot_then_full.{log,status}` launcher record
because that is useful to remember how the final Phase 3 smoke benchmark was
started without keeping a separate top-level log directory.
