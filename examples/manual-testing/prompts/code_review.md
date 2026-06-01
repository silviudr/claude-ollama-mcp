Review the following Python code for bugs, security issues, and improvements:

```python
import sqlite3
import os

def get_user(db_path, username):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    result = cursor.fetchone()
    return result

def read_config(path):
    with open(path) as f:
        return eval(f.read())

def process_items(items):
    results = []
    for i in range(0, len(items)):
        if items[i] != None:
            results.append(items[i] * 2)
    return results

def divide(a, b):
    return a / b
```
