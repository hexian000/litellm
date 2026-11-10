from __future__ import annotations

import asyncio

import aiohttp
import httpx
from aiohttp.abc import AbstractStreamWriter
from aiohttp.payload import BytesPayload
from pydantic import BaseModel
from typing_extensions import override


class _Timeouts(BaseModel):
    connect: float | None = None
    read: float | None = None
    write: float | None = None
    pool: float | None = None


class _WriteBody(BytesPayload):
    def __init__(self, body: bytes, timeout: float | None) -> None:
        super().__init__(body)
        self.timeout = timeout

    @override
    async def write(self, writer: AbstractStreamWriter) -> None:
        async with asyncio.timeout(self.timeout):
            await super().write(writer)

    @override
    async def write_with_length(self, writer: AbstractStreamWriter, content_length: int | None) -> None:
        async with asyncio.timeout(self.timeout):
            await super().write_with_length(writer, content_length)


class PrismaEngineTransport(httpx.AsyncBaseTransport):
    def __init__(self, max_connections: int = 1000) -> None:
        self._slots = asyncio.BoundedSemaphore(max_connections)
        self._keepalive_slots = asyncio.BoundedSemaphore(min(20, max_connections))
        self._keepalive_limit = min(20, max_connections)
        self._sessions: tuple[aiohttp.ClientSession, aiohttp.ClientSession] | None = None
        self._origin: tuple[str, int] | None = None
        self._generation = 0
        self._fallback = httpx.AsyncHTTPTransport(
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=0)
        )

    def _clients(self) -> tuple[aiohttp.ClientSession, aiohttp.ClientSession]:
        if self._sessions is None:
            self._sessions = (self._client(True), self._client(False))
        return self._sessions

    def _client(self, keepalive: bool) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self._keepalive_limit if keepalive else 0, force_close=not keepalive),
            cookie_jar=aiohttp.DummyCookieJar(),
            auto_decompress=False,
            trust_env=False,
            max_line_size=65536,
            max_field_size=65536,
        )

    @override
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        generation = self._generation
        timeouts = (
            _Timeouts.model_validate(request.extensions["timeout"]) if "timeout" in request.extensions else _Timeouts()
        )
        try:
            async with asyncio.timeout(timeouts.pool):
                await self._slots.acquire()
        except TimeoutError as error:
            raise httpx.PoolTimeout("Prisma engine HTTP connection limit reached", request=request) from error
        try:
            if generation != self._generation:
                raise httpx.TransportError("Prisma engine HTTP transport is closed", request=request)
            if (
                request.url.scheme != "http"
                or request.url.host not in ("127.0.0.1", "localhost", "::1")
                or not isinstance(request.stream, httpx.ByteStream)
                or "transfer-encoding" in request.headers
                or any(not key.isascii() or not value.isascii() for key, value in request.headers.raw)
            ):
                return await self._fallback.handle_async_request(request)
            origin = (request.url.host, request.url.port or 80)
            if self._origin is None:
                self._origin = origin
            if self._origin != origin:
                return await self._fallback.handle_async_request(request)
            return await self._request(request, timeouts)
        finally:
            self._slots.release()

    async def _request(self, request: httpx.Request, timeouts: _Timeouts) -> httpx.Response:
        keepalive = not self._keepalive_slots.locked()
        if keepalive:
            await self._keepalive_slots.acquire()
        try:
            return await self._send(request, timeouts, self._clients()[0 if keepalive else 1])
        finally:
            if keepalive:
                self._keepalive_slots.release()

    async def _send(self, request: httpx.Request, timeouts: _Timeouts, client: aiohttp.ClientSession) -> httpx.Response:
        body = await request.aread()
        if request.headers.get("content-length", str(len(body))) != str(len(body)):
            return await self._fallback.handle_async_request(request)
        try:
            async with client.request(
                request.method,
                str(request.url),
                headers=tuple((key.decode("ascii"), value.decode("ascii")) for key, value in request.headers.raw),
                data=_WriteBody(body, timeouts.write) if body else None,
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=None, connect=timeouts.connect, sock_read=timeouts.read),
            ) as response:
                content = await response.read()
                return httpx.Response(response.status, headers=response.raw_headers, content=content)
        except aiohttp.ConnectionTimeoutError as error:
            raise httpx.ConnectTimeout(str(error), request=request) from error
        except aiohttp.SocketTimeoutError as error:
            raise httpx.ReadTimeout(str(error), request=request) from error
        except TimeoutError as error:
            raise httpx.WriteTimeout(str(error), request=request) from error
        except aiohttp.ClientConnectorError as error:
            raise httpx.ConnectError(str(error), request=request) from error
        except aiohttp.ClientResponseError as error:
            raise httpx.RemoteProtocolError(str(error), request=request) from error
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as error:
            raise httpx.ReadError(str(error), request=request) from error

    @override
    async def aclose(self) -> None:
        self._generation += 1
        sessions = self._sessions
        self._sessions = None
        self._origin = None
        try:
            if sessions is not None:
                await asyncio.gather(*(session.close() for session in sessions))
        finally:
            await self._fallback.aclose()
