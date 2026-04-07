from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable


KernelFn = Callable[..., Any]


@dataclass(slots=True)
class KernelImplementation:
    backends: dict[str, KernelFn] = field(default_factory=dict)


@dataclass(slots=True)
class BackendProvider:
    backend_id: str
    description: str
    aliases: tuple[str, ...] = ()
    availability_probe: Callable[[], bool] | None = None
    unavailable_reason: str = ""

    def available(self) -> bool:
        if self.availability_probe is None:
            return True
        try:
            return bool(self.availability_probe())
        except Exception:
            return False


@dataclass(slots=True)
class ResolvedKernel:
    name: str
    backend_id: str
    fn: KernelFn


class KernelResolutionError(RuntimeError):
    """Raised when no registered implementation can satisfy the requested backend."""


class KernelRegistry:
    def __init__(self) -> None:
        self._impls: dict[str, KernelImplementation] = {}
        self._providers: dict[str, BackendProvider] = {}
        self._aliases: dict[str, str] = {}

    def register_provider(
        self,
        backend_id: str,
        *,
        description: str,
        aliases: Sequence[str] = (),
        availability_probe: Callable[[], bool] | None = None,
        unavailable_reason: str = "",
    ) -> None:
        canonical = self.canonical_backend(backend_id)
        provider = BackendProvider(
            backend_id=canonical,
            description=description,
            aliases=tuple(str(alias).strip().lower() for alias in aliases),
            availability_probe=availability_probe,
            unavailable_reason=unavailable_reason,
        )
        self._providers[canonical] = provider
        self._aliases[canonical] = canonical
        for alias in provider.aliases:
            self._aliases[alias] = canonical

    def canonical_backend(self, backend_id: str) -> str:
        normalized = str(backend_id).strip().lower()
        return self._aliases.get(normalized, normalized)

    def register(
        self,
        name: str,
        cpu: KernelFn | None = None,
        gpu: KernelFn | None = None,
        native: KernelFn | None = None,
        **implementations: KernelFn | None,
    ) -> None:
        impls = self._impls.setdefault(name, KernelImplementation())
        merged: dict[str, KernelFn | None] = dict(implementations)
        if cpu is not None:
            merged["python"] = cpu
        if gpu is not None:
            merged["cuda"] = gpu
        if native is not None:
            merged["native"] = native
        for backend_id, fn in merged.items():
            if fn is None:
                continue
            impls.backends[self.canonical_backend(backend_id)] = fn

    def implementations(self, name: str) -> dict[str, KernelFn]:
        if name not in self._impls:
            raise KeyError(f"Kernel '{name}' is not registered")
        return dict(self._impls[name].backends)

    def supported_backends(self, name: str, *, available_only: bool = False) -> list[str]:
        impls = self.implementations(name)
        out: list[str] = []
        for backend_id in sorted(impls):
            provider = self._providers.get(backend_id)
            if available_only and provider is not None and not provider.available():
                continue
            out.append(backend_id)
        return out

    def resolve(
        self,
        name: str,
        *,
        requested: str = "auto",
        fallback_order: Sequence[str] = (),
        strict_requested: bool = False,
        implementations: Mapping[str, KernelFn] | None = None,
    ) -> ResolvedKernel:
        impls = dict(implementations) if implementations is not None else self.implementations(name)
        normalized_requested = self.canonical_backend(requested)
        order: list[str] = []
        if normalized_requested != "auto":
            order.append(normalized_requested)
        for backend_id in fallback_order:
            canonical = self.canonical_backend(backend_id)
            if canonical not in order:
                order.append(canonical)
        if normalized_requested == "auto" and not order:
            raise KernelResolutionError(f"Kernel '{name}' cannot resolve backend 'auto' without a fallback order")

        request_error: str | None = None
        for backend_id in order:
            fn = impls.get(backend_id)
            if fn is None:
                if backend_id == normalized_requested:
                    request_error = f"Kernel '{name}' has no '{backend_id}' implementation"
                continue
            provider = self._providers.get(backend_id)
            if provider is not None and not provider.available():
                reason = provider.unavailable_reason or f"Backend '{backend_id}' is unavailable"
                if backend_id == normalized_requested:
                    request_error = reason
                continue
            return ResolvedKernel(name=name, backend_id=backend_id, fn=fn)

        if request_error is not None and strict_requested:
            raise KernelResolutionError(request_error)
        if request_error is not None:
            raise KernelResolutionError(request_error)
        raise KernelResolutionError(f"Kernel '{name}' has no implementation for requested order {tuple(order)}")

    def get(self, name: str, backend: str = "auto") -> KernelFn:
        normalized = self.canonical_backend(backend)
        if normalized == "native":
            return self.resolve(name, requested="native", fallback_order=("python",)).fn
        if normalized == "cuda":
            return self.resolve(name, requested="cuda", fallback_order=("python",)).fn
        return self.resolve(name, requested="auto", fallback_order=("python",)).fn

    def coverage_manifest(self) -> dict[str, Any]:
        providers = {
            backend_id: {
                "description": provider.description,
                "aliases": list(provider.aliases),
                "available": provider.available(),
                "unavailable_reason": provider.unavailable_reason,
            }
            for backend_id, provider in sorted(self._providers.items())
        }
        kernels = {
            kernel_name: {
                "supported_backends": self.supported_backends(kernel_name),
                "available_backends": self.supported_backends(kernel_name, available_only=True),
                "baseline_backend": "python" if "python" in impl.backends else None,
            }
            for kernel_name, impl in sorted(self._impls.items())
        }
        return {"providers": providers, "kernels": kernels}


DEFAULT_REGISTRY = KernelRegistry()
