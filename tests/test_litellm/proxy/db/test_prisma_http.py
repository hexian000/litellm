import asyncio
import gzip
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio
from aiohttp import web
from prisma._async_http import HTTP
from pydantic import TypeAdapter

from litellm.proxy.db.prisma_client import default_prisma_http_config
from litellm.proxy.db.prisma_http import PrismaEngineTransport


@dataclass(frozen=True)
class Server:
    url: str
    requests: asyncio.Queue[tuple[str, bytes]]
    entered: asyncio.Event
    release: asyncio.Event
    connections: asyncio.Queue[asyncio.Transport | None]


@pytest_asyncio.fixture(loop_scope="function")
async def server() -> AsyncIterator[Server]:
    requests: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()
    connections: asyncio.Queue[asyncio.Transport | None] = asyncio.Queue()
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request: web.Request) -> web.Response:
        body = await request.read()
        requests.put_nowait((request.method, body))
        connections.put_nowait(request.transport)
        if request.path == "/wait":
            entered.set()
            await release.wait()
        if request.path == "/gzip":
            return web.Response(body=gzip.compress(b'{"ok":true}'), headers={"Content-Encoding": "gzip"})
        if request.path == "/redirect":
            return web.Response(status=302, headers={"Location": "/echo"})
        if request.path == "/error":
            return web.json_response({"errors": [{"error": "retained"}]}, status=500)
        return web.Response(body=body or b'{"ok":true}')

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        host, port = TypeAdapter(tuple[str, int]).validate_python(runner.addresses[0])
        yield Server(f"http://{host}:{port}", requests, entered, release, connections)
    finally:
        release.set()
        await runner.cleanup()


async def test_default_config_preserves_httpx_response_contract(server: Server) -> None:
    engine_http = HTTP(**default_prisma_http_config())
    engine_http.open()
    async with engine_http.session as client:
        response = await client.post(server.url + "/echo", content=b'{"sql":"SELECT 1"}')
        assert response.status_code == 200
        assert response.json() == {"sql": "SELECT 1"}
        assert response.request.method == "POST"
        assert response.is_closed
        assert await server.requests.get() == ("POST", b'{"sql":"SELECT 1"}')
        error = await client.get(server.url + "/error")
        assert error.status_code == 500 and error.json() == {"errors": [{"error": "retained"}]}
        compressed = await client.get(server.url + "/gzip")
        assert compressed.json() == {"ok": True}
        redirect = await client.get(server.url + "/redirect")
        assert redirect.status_code == 302
        followed = await client.get(server.url + "/redirect", follow_redirects=True)
        assert len(followed.history) == 1 and followed.json() == {"ok": True}


async def test_pool_timeout_and_cancelled_waiter_do_not_take_capacity(server: Server) -> None:
    async with httpx.AsyncClient(transport=PrismaEngineTransport(max_connections=1), timeout=2) as client:
        first = asyncio.create_task(client.get(server.url + "/wait"))
        await server.entered.wait()
        with pytest.raises(httpx.PoolTimeout):
            await client.get(server.url + "/echo", timeout=httpx.Timeout(1, pool=0.01))
        cancelled = asyncio.create_task(client.get(server.url + "/echo"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        server.release.set()
        assert (await first).status_code == 200
        assert (await client.get(server.url + "/echo")).status_code == 200


async def test_read_timeout_is_an_httpx_timeout(server: Server) -> None:
    async with httpx.AsyncClient(transport=PrismaEngineTransport()) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.get(server.url + "/wait", timeout=httpx.Timeout(1, read=0.01))
        server.release.set()
        assert (await client.get(server.url + "/echo")).status_code == 200


async def test_close_interrupts_live_requests(server: Server) -> None:
    transport = PrismaEngineTransport(max_connections=1)
    client = httpx.AsyncClient(transport=transport, timeout=2)
    task = asyncio.create_task(client.get(server.url + "/wait"))
    await server.entered.wait()
    queued = asyncio.create_task(client.get(server.url + "/echo"))
    await asyncio.sleep(0)
    await client.aclose()
    with pytest.raises(httpx.TransportError):
        await task
    with pytest.raises(httpx.TransportError):
        await queued
    assert server.requests.qsize() == 1
    async with httpx.AsyncClient(transport=transport) as replacement:
        assert (await replacement.get(server.url + "/echo")).status_code == 200


async def test_refused_connection_keeps_httpx_error_type(server: Server) -> None:
    host = server.url.removeprefix("http://").split(":")[0]
    listener = await asyncio.start_server(lambda _reader, _writer: None, host, 0)
    address = TypeAdapter(tuple[str, int]).validate_python(listener.sockets[0].getsockname())
    listener.close()
    await listener.wait_closed()
    async with httpx.AsyncClient(transport=PrismaEngineTransport()) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(f"http://{host}:{address[1]}", timeout=0.2)


async def test_different_loopback_origins_remain_supported(server: Server) -> None:
    async with httpx.AsyncClient(transport=PrismaEngineTransport()) as client:
        first = await client.get(server.url + "/echo")
        second = await client.get(server.url.replace("127.0.0.1", "localhost") + "/echo")
        assert first.json() == second.json() == {"ok": True}


async def test_concurrent_queries_do_not_serialize_at_keepalive_limit(server: Server) -> None:
    async with httpx.AsyncClient(transport=PrismaEngineTransport(), timeout=2) as client:
        tasks = tuple(asyncio.create_task(client.get(server.url + "/wait")) for _ in range(32))
        for _ in range(200):
            if server.requests.qsize() == 32:
                break
            await asyncio.sleep(0.005)
        assert server.requests.qsize() == 32
        server.release.set()
        responses = await asyncio.gather(*tasks)
        assert all(response.status_code == 200 for response in responses)
        await asyncio.sleep(0.05)
        connections = frozenset(server.connections.get_nowait() for _ in range(server.connections.qsize()))
        assert sum(connection is not None and not connection.is_closing() for connection in connections) <= 20


async def test_streaming_request_uses_httpx_framing(server: Server) -> None:
    async def body() -> AsyncIterator[bytes]:
        yield b'{"sql":'
        yield b'"SELECT 1"}'

    async with httpx.AsyncClient(transport=PrismaEngineTransport()) as client:
        response = await client.post(server.url + "/echo", content=body())
        assert response.json() == {"sql": "SELECT 1"}


async def test_write_timeout_is_an_httpx_timeout() -> None:
    release = asyncio.Event()
    done = asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()
            done.set()

    listener = await asyncio.start_server(handler, "127.0.0.1", 0, limit=1024)
    _, port = TypeAdapter(tuple[str, int]).validate_python(listener.sockets[0].getsockname())
    try:
        async with httpx.AsyncClient(transport=PrismaEngineTransport()) as client:
            with pytest.raises(httpx.WriteTimeout):
                await client.post(
                    f"http://127.0.0.1:{port}",
                    content=b"x" * (16 * 1024 * 1024),
                    timeout=httpx.Timeout(1, write=0.01),
                )
    finally:
        release.set()
        listener.close()
        await listener.wait_closed()
        await asyncio.wait_for(done.wait(), 1)
