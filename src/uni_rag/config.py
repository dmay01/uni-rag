"""Validated TOML configuration; all relative paths are relative to the config file."""

import ipaddress
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = (".md", ".txt", ".pdf", ".pptx", ".docx", ".tex")


@dataclass(frozen=True)
class Source:
    name: str
    path: Path


@dataclass(frozen=True)
class Config:
    sources: tuple[Source, ...]
    data_dir: Path = PROJECT_ROOT / "data"
    include_extensions: tuple[str, ...] = EXTENSIONS
    exclude_patterns: tuple[str, ...] = ()
    max_file_mb: float = 50
    skip_cloud_only: bool = True
    embedding_batch_size: int = 16
    target_tokens: int = 500
    overlap_fraction: float = 0.15
    min_slide_tokens: int = 30
    base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    chat_model: str = "qwen3:14b"
    chat_models: tuple[str, ...] = ("qwen3:4b-q4_K_M", "qwen3:14b", "qwen3:30b-a3b-q4_K_M")
    timeout_seconds: float = 180
    context_window: int = 8192
    num_predict: int = 512
    answer_max_words: int = 150
    top_k: int = 8
    candidate_k: int = 32
    rrf_constant: int = 60
    energy_backend: str = "auto"
    power_log: Path = PROJECT_ROOT / "data/power.log"
    baseline_seconds: float = 3
    sample_interval_ms: int = 200
    max_sample_gap_ms: int = 1000
    sample_wait_seconds: float = 6
    root: Path = field(default=PROJECT_ROOT, repr=False)

    def __post_init__(self) -> None:
        names = [source.name for source in self.sources]
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("Configure at least one source with unique, nonempty names")
        url = urlparse(self.base_url)
        try:
            loopback = ipaddress.ip_address(url.hostname or "").is_loopback
        except ValueError:
            loopback = url.hostname == "localhost"
        if url.scheme != "http" or not loopback or url.username or url.password:
            raise ValueError("Ollama must use HTTP on localhost or a loopback IP address")
        if url.path not in ("", "/") or url.query or url.fragment:
            raise ValueError("Ollama base_url must not include a path, query or fragment")
        if self.target_tokens < 16 or not 0 <= self.overlap_fraction < 1:
            raise ValueError("target_tokens must be >=16 and overlap_fraction in [0, 1)")
        for name in ("embedding_batch_size", "top_k", "candidate_k", "rrf_constant"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_file_mb <= 0 or self.timeout_seconds <= 0 or self.min_slide_tokens < 0:
            raise ValueError("Invalid size, timeout or minimum slide size")
        if self.context_window < 1024:
            raise ValueError("context_window must be at least 1024 tokens")
        if self.num_predict < 1 or self.answer_max_words < 1:
            raise ValueError("num_predict and answer_max_words must be positive")
        if self.max_sample_gap_ms < self.sample_interval_ms or self.sample_wait_seconds <= 0:
            raise ValueError("Invalid maximum power-sample gap or wait timeout")
        if self.sample_interval_ms < 1 or self.baseline_seconds < 0:
            raise ValueError("Invalid energy sampling settings")
        if self.energy_backend not in {"auto", "macmon", "powermetrics", "none"}:
            raise ValueError("energy_backend must be auto/macmon/powermetrics/none")
        for path in (self.data_dir, self.power_log):
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("All generated data and power logs must stay inside the project")
        for source in self.sources:
            if self.data_dir.resolve().is_relative_to(source.path.resolve()):
                raise ValueError("The data directory must not be inside a source directory")

    def selected_sources(self, name: str | None = None) -> tuple[Source, ...]:
        selected = tuple(s for s in self.sources if name is None or s.name == name)
        if not selected:
            raise ValueError(f"Unknown source: {name}")
        return selected


def load_config(path: Path = PROJECT_ROOT / "config.toml") -> Config:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    base = path.resolve().parent

    def local_path(value: str) -> Path:
        expanded = Path(value).expanduser()
        return expanded if expanded.is_absolute() else base / expanded

    values: dict = {"root": PROJECT_ROOT}
    allowed = set(Config.__dataclass_fields__) - {"sources", "root"}
    for section in ("ingest", "chunking", "ollama", "retrieval", "storage", "energy"):
        for key, value in raw.get(section, {}).items():
            if key not in allowed:
                raise ValueError(f"Unknown configuration key: {section}.{key}")
            if key in {"data_dir", "power_log"}:
                value = local_path(value)
            elif key in {"include_extensions", "exclude_patterns", "chat_models"}:
                value = tuple(value)
            values[key] = value
    sources = tuple(Source(s["name"], local_path(s["path"])) for s in raw.get("sources", []))
    return Config(sources=sources, **values)
