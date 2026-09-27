"""A pooled connection whose server messages asyncpg cannot decode is discarded and
the operation retried on a fresh one.

Apache AGE with a corrupted per-backend label cache answers ``DETACH DELETE`` with
``relation "<graph>.<garbage>" does not exist`` whose garbage is not UTF-8; asyncpg
then raises ``UnicodeDecodeError`` from its protocol decoder and the PostgresError is
lost. The fault follows the backend, so ``_run_with_retry`` terminates that one
connection and retries, without resetting the whole pool."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from lightrag.kg.postgres_impl import (
    TRANSIENT_DB_EXCEPTIONS,
    PoisonedConnectionError,
    PostgreSQLDB,
)


def _raise_from_an_asyncpg_frame() -> None:
    # Emulate asyncpg's protocol decoder: the raising frame's file lives under asyncpg/.
    code = compile(
        "raise UnicodeDecodeError('utf-8', b'\\x9b', 0, 1, 'invalid start byte')",
        "asyncpg/protocol/coreproto.pyx",
        "exec",
    )
    exec(code, {})


class _FakeConnection:
    def __init__(self, *, poisoned: bool = False, blip: bool = False) -> None:
        self.poisoned = poisoned
        self.blip = blip
        self.terminated = False
        self.executed: list[str] = []

    async def execute(self, sql: str) -> str:
        self.executed.append(sql)
        if self.poisoned:
            _raise_from_an_asyncpg_frame()
        if self.blip:
            raise ConnectionResetError("connection reset by peer")
        return "DELETE 18"

    def terminate(self) -> None:
        self.terminated = True

    def is_closed(self) -> bool:
        return self.terminated


class _Acquire:
    def __init__(self, conn: _FakeConnection) -> None:
        self.conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakePool:
    def __init__(self, conns: list[_FakeConnection]) -> None:
        self.conns = list(conns)
        self.acquired: list[_FakeConnection] = []

    def acquire(self) -> _Acquire:
        conn = self.conns.pop(0)
        self.acquired.append(conn)
        return _Acquire(conn)


def make_db(pool: _FakePool) -> PostgreSQLDB:
    db = PostgreSQLDB.__new__(PostgreSQLDB)  # skip __init__/connection setup
    db.pool = pool
    db._pool_reconnect_lock = asyncio.Lock()
    db._transient_exceptions = TRANSIENT_DB_EXCEPTIONS
    db.connection_retry_attempts = 3
    db.connection_retry_backoff = 0
    db.connection_retry_backoff_max = 0
    db.pool_close_timeout = 1.0
    db._reset_pool = AsyncMock()
    return db


async def _detach_delete(conn: _FakeConnection) -> str:
    return await conn.execute(
        "MATCH (n:base) WHERE n.entity_id IN [...] DETACH DELETE n"
    )


@pytest.mark.offline
def test_poisoned_connection_error_is_retryable():
    assert issubclass(PoisonedConnectionError, TRANSIENT_DB_EXCEPTIONS)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_poisoned_connection_is_terminated_and_the_operation_retried_on_a_fresh_one():
    bad, good = _FakeConnection(poisoned=True), _FakeConnection()
    pool = _FakePool([bad, good])
    db = make_db(pool)

    assert await db._run_with_retry(_detach_delete) == "DELETE 18"

    assert bad.terminated and not good.terminated
    assert pool.acquired == [bad, good]
    assert good.executed == bad.executed  # the same statement, replayed as-is
    db._reset_pool.assert_not_awaited()  # only the poisoned connection was dropped


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_decode_error_from_our_own_code_is_not_treated_as_poison():
    conn = _FakeConnection()
    pool = _FakePool([conn, conn, conn])
    db = make_db(pool)

    async def decode_our_own_bytes(_conn: _FakeConnection) -> str:
        return b"\x9b".decode("utf-8")  # raised from this file, not from asyncpg

    with pytest.raises(UnicodeDecodeError):
        await db._run_with_retry(decode_our_own_bytes)

    assert not conn.terminated
    assert pool.acquired == [conn]  # no retry: not a connection fault


@pytest.mark.offline
@pytest.mark.asyncio
async def test_poison_on_every_attempt_surfaces_as_poisoned_connection_error():
    conns = [_FakeConnection(poisoned=True) for _ in range(3)]
    pool = _FakePool(conns)
    db = make_db(pool)

    with pytest.raises(PoisonedConnectionError):
        await db._run_with_retry(_detach_delete)

    assert all(c.terminated for c in conns)
    assert len(pool.acquired) == 3  # connection_retry_attempts
    db._reset_pool.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_an_ordinary_transient_error_still_resets_the_pool():
    flaky, good = _FakeConnection(blip=True), _FakeConnection()
    pool = _FakePool([flaky, good])
    db = make_db(pool)

    assert await db._run_with_retry(_detach_delete) == "DELETE 18"

    assert not flaky.terminated  # unchanged behaviour for ordinary blips
    db._reset_pool.assert_awaited_once()
