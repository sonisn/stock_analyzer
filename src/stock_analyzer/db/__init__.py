"""SQLAlchemy 2.x persistence layer.

session.py:      engine + get_session() contextmanager
tables.py:       table classes (Run, Candidate, Pick, ...)
repository.py:   CRUD repository functions (insert_run, etc.)
track_record.py: read-only analytics queries for return calculation
"""

from __future__ import annotations
