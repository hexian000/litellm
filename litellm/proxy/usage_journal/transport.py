from __future__ import annotations

import zlib
from collections.abc import AsyncIterator, Callable

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import override

from .lifecycle import OBJECTS, AttemptHandle, WireUsage, current_attempt, usage_from
from .protocol import UnknownUsage

TEXT = TypeAdapter(str)


def decoder(encoding: str) -> Callable[[bytes], bytes] | None:
    if encoding in ("", "identity"):
        return lambda value: value
    if encoding == "gzip":
        return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress
    if encoding == "deflate":
        return zlib.decompressobj().decompress
    return None


class SupplierStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, attempt: AttemptHandle, decode: Callable[[bytes], bytes]) -> None:
        self.stream = stream
        self.attempt = attempt
        self.decode = decode
        self.wire = WireUsage()
        self.valid = True

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.stream:
            if self.valid:
                try:
                    self.wire.feed(self.decode(chunk))
                    self.attempt.supplier_usage = self.wire.result()
                except zlib.error:
                    self.valid = False
                    self.attempt.supplier_usage = UnknownUsage(state="unknown", reason="supplier_stream_encoding")
            yield chunk

    @override
    async def aclose(self) -> None:
        await self.stream.aclose()


async def capture_provider_response(response: httpx.Response) -> None:
    attempt = current_attempt.get()
    if (
        attempt is None
        or not attempt.called
        or attempt.supplier_response_seen
        or response.request.method != "POST"
        or not response.is_success
    ):
        return
    attempt.supplier_response_seen = True
    if "text/event-stream" in TEXT.validate_python(response.headers.get("content-type", "")).lower():
        decode = decoder(TEXT.validate_python(response.headers.get("content-encoding", "")).strip().lower())
        if decode is not None and isinstance(response.stream, httpx.AsyncByteStream):
            response.stream = SupplierStream(response.stream, attempt, decode)
        return
    await response.aread()
    try:
        attempt.supplier_usage = usage_from(OBJECTS.validate_json(response.content))
    except (ValueError, ValidationError):
        attempt.supplier_usage = UnknownUsage(state="unknown", reason="supplier_response_invalid")
