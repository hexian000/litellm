from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


class IncomingRequest(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class IncomingHeaders:
    headers: Mapping[str, str]
