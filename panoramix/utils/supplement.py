import hashlib
import json
import logging
import lzma
import os
import sqlite3
from pathlib import Path
from typing import Optional
from panoramix.utils.helpers import (
    cache_dir,
    cached,
)

"""
    a module for management of bytes4 signatures from the database

    data/abi_dump.xz has one json object per line:
        {"selector": "0xa9059cbb", "abi": {"name": "transfer", "inputs": [...], ...}}

    It is loaded into an sqlite database in the cache directory:
        abi(selector TEXT PRIMARY KEY, abi TEXT) - abi is the json of the "abi" field
    and loaded again whenever the dump changes (its user_version tells which one it has).

"""

logger = logging.getLogger(__name__)


def abi_path():
    return cache_dir() / "abi_db.sqlite3"


def dump_path():
    return Path(__file__).parent.parent / "data" / "abi_dump.xz"


@cached
def dump_version() -> int:
    """A number that identifies the dump, from 1 to 2^31-1 to fit in user_version."""
    sha = hashlib.sha256()
    with open(dump_path(), "rb") as f:
        while chunk := f.read(1 << 20):
            sha.update(chunk)
    return int.from_bytes(sha.digest()[:4], "big") % (2**31 - 1) + 1


def db_version() -> Optional[int]:
    try:
        db = sqlite3.connect(abi_path().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error:  # no database yet
        return None
    try:
        return db.execute("PRAGMA user_version").fetchall()[0][0]
    except sqlite3.Error:  # not an sqlite file
        return None
    finally:
        db.close()


def check_supplements():
    if db_version() == dump_version():
        return

    logger.info("Loading %s into %s...", dump_path(), abi_path())

    # The database is built next to its final place and moved there once it's
    # complete: an interrupted load (timeout, ^C...) must not leave behind a
    # partial database that would be taken for a complete one from then on.
    tmp_path = abi_path().with_name(f"{abi_path().name}.{os.getpid()}.tmp")
    try:
        # A killed process may have left a partial one with our pid.
        tmp_path.unlink(missing_ok=True)
        db = sqlite3.connect(str(tmp_path))
        try:
            # No journal: if anything goes wrong, the whole file is thrown away.
            db.executescript(
                "PRAGMA journal_mode = OFF;"
                f"PRAGMA user_version = {dump_version()};"
                "CREATE TABLE abi (selector TEXT PRIMARY KEY, abi TEXT NOT NULL)"
                " WITHOUT ROWID;"
            )
            with lzma.open(dump_path()) as inf:
                entries = (json.loads(line) for line in inf)
                db.executemany(
                    "INSERT OR REPLACE INTO abi VALUES (?, ?)",
                    ((e["selector"], json.dumps(e["abi"])) for e in entries),
                )
            db.commit()
        finally:
            db.close()
        os.replace(tmp_path, abi_path())
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info("%s is ready.", abi_path())


@cached
def fetch_sig(hash) -> Optional[dict]:
    check_supplements()

    if type(hash) == str:
        hash = int(hash, 16)
    hash = "{:#010x}".format(hash)

    # Read-only, so that a missing file is an error instead of a new, empty database.
    db = sqlite3.connect(abi_path().as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT abi FROM abi WHERE selector = ?", (hash,)).fetchall()
    finally:
        db.close()

    if not rows:
        return None

    abi = json.loads(rows[0][0])
    # A quarter of the dump is what a decompiler guessed of functions whose
    # signature wasn't known - unknowna8b0c2ca(uint64 _param1) for a function
    # of an address - not signatures: one is only where it hashes to the
    # selector.
    return abi if hashes_to(abi, hash) else None


def hashes_to(abi, selector):
    """If the signature of abi hashes to the selector (0x and its 8 digits)."""
    from eth_hash.auto import keccak

    from panoramix.utils.signatures import canonical_type

    text = "{}({})".format(
        abi.get("name", ""),
        ",".join(
            canonical_type(i["type"], i.get("components"))
            for i in abi.get("inputs", [])
        ),
    )
    # (the selectors of the libraries of old compilers hash a storage pointer
    # without its space, `getMin(uint32[]storage)`)
    return any(
        "0x" + keccak(t.encode())[:4].hex() == selector
        for t in (text, text.replace(" ", ""))
    )
