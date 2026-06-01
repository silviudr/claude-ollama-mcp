Write a Python function called `parse_duration` that converts a human-readable
duration string into total seconds.

Supported formats:
- "30s" → 30
- "5m" → 300
- "2h" → 7200
- "1h30m" → 5400
- "2h15m30s" → 8130

Requirements:
- Accept any combination of hours (h), minutes (m), seconds (s)
- Return an integer
- Raise ValueError for invalid input
- No external dependencies
