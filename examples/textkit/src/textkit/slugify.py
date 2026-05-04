import unicodedata
import re


def slugify(s: str) -> str:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[^a-z0-9]+', '-', s.lower())
    return s.strip('-')
