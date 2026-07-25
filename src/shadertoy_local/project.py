"""Project discovery and configuration.

A Shadertoy project is just a folder. Conventional layout::

    myshader/
      shadertoy.json      # optional; channel wiring and defaults
      common.glsl         # optional; prepended to every pass
      image.glsl          # required; the final pass
      buffer_a.glsl       # optional; Buffer A .. Buffer D
      textures/wood.png
      golden/             # reference images for `shadertoy test`

Nothing but ``image.glsl`` is mandatory, so a one-file shader works with zero
configuration. ``shadertoy.json`` is the canonical config format; an equivalent
``shadertoy.toml`` is also accepted since the schema is identical.

Config schema::

    {
      "defaults": { "width": 640, "height": 360, "fps": 60 },
      "image": {
        "channels": {
          "0": { "type": "buffer",   "source": "buffer_a" },
          "1": { "type": "texture",  "source": "textures/wood.png",
                 "filter": "linear", "wrap": "repeat", "vflip": true },
          "2": { "type": "builtin",  "source": "noise" },
          "3": { "type": "keyboard" }
        }
      },
      "buffer_a": { "scale": 0.5, "channels": { "0": "buffer_a" } }
    }

``type`` is optional and inferred from ``source``; when present it is validated,
which turns a typo into a clear error instead of a confusing missing-file.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Buffer pass keys, in Shadertoy's fixed execution order.
BUFFER_NAMES = ("buffer_a", "buffer_b", "buffer_c", "buffer_d")
#: Every pass key, in execution order. The image pass always runs last.
PASS_NAMES = (*BUFFER_NAMES, "image")

#: Filenames accepted for each pass, in priority order.
_PASS_FILENAMES: dict[str, tuple[str, ...]] = {
    "image": ("image.glsl", "image.frag", "main.glsl", "shader.glsl"),
    "buffer_a": ("buffer_a.glsl", "bufferA.glsl", "buffer_a.frag"),
    "buffer_b": ("buffer_b.glsl", "bufferB.glsl", "buffer_b.frag"),
    "buffer_c": ("buffer_c.glsl", "bufferC.glsl", "buffer_c.frag"),
    "buffer_d": ("buffer_d.glsl", "bufferD.glsl", "buffer_d.frag"),
}
_COMMON_FILENAMES = ("common.glsl", "common.frag")

CONFIG_NAMES = (
    "shadertoy.json",
    ".shadertoy.json",
    "shadertoy.toml",
    ".shadertoy.toml",
)

#: Procedural channel sources that need no asset files. Keeping these built in
#: means projects stay self-contained and tests stay reproducible; Shadertoy's
#: own texture assets cannot be redistributed.
BUILTIN_TEXTURES = (
    "noise",
    "rgba-noise",
    "gray-noise",
    "blue-noise",
    "checker",
    "uv",
    "gradient",
    "white",
    "black",
    "keyboard",
)

_FILTERS = ("nearest", "linear", "mipmap")
_WRAPS = ("clamp", "repeat")

#: Channel kinds. Inferred from ``source`` when not stated explicitly.
CHANNEL_TYPES = ("buffer", "texture", "builtin", "keyboard")


class ProjectError(RuntimeError):
    """Raised for malformed projects or configuration."""


@dataclass
class ChannelBinding:
    """One ``iChannelN`` input for one pass."""

    source: str
    kind: str = "texture"
    filter: str = "linear"
    wrap: str = "repeat"
    vflip: bool = True
    #: Resolved asset path for file-backed sources.
    path: Path | None = None

    @property
    def is_buffer(self) -> bool:
        return self.kind == "buffer"

    @property
    def is_builtin(self) -> bool:
        return self.kind in ("builtin", "keyboard")

    @property
    def is_keyboard(self) -> bool:
        return self.kind == "keyboard"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "source": self.source,
            "filter": self.filter,
            "wrap": self.wrap,
            "vflip": self.vflip,
            "path": str(self.path) if self.path else None,
        }


@dataclass
class PassSpec:
    """A single render pass."""

    name: str
    path: Path
    source: str
    channels: dict[int, ChannelBinding] = field(default_factory=dict)
    #: Buffer passes render at a fraction of the output size when set.
    scale: float = 1.0

    @property
    def is_image(self) -> bool:
        return self.name == "image"

    @property
    def label(self) -> str:
        if self.name == "image":
            return "Image"
        return "Buffer " + self.name.rsplit("_", 1)[1].upper()


@dataclass
class Project:
    """A fully resolved, ready-to-render project."""

    root: Path
    passes: dict[str, PassSpec]
    common: str | None = None
    common_path: Path | None = None
    config_path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def ordered_passes(self) -> list[PassSpec]:
        """Passes in execution order: Buffer A-D, then Image."""
        return [self.passes[n] for n in PASS_NAMES if n in self.passes]

    @property
    def buffer_passes(self) -> list[PassSpec]:
        return [self.passes[n] for n in BUFFER_NAMES if n in self.passes]

    def source_files(self) -> list[Path]:
        files = [p.path for p in self.ordered_passes]
        if self.common_path:
            files.append(self.common_path)
        return files

    def default(self, key: str, fallback: Any) -> Any:
        """Read a value from the config's ``[defaults]`` table."""
        return self.config.get("defaults", {}).get(key, fallback)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "config": str(self.config_path) if self.config_path else None,
            "common": str(self.common_path) if self.common_path else None,
            "passes": {
                p.name: {
                    "label": p.label,
                    "file": str(p.path.relative_to(self.root)),
                    "scale": p.scale,
                    "channels": {
                        str(i): b.to_dict() for i, b in sorted(p.channels.items())
                    },
                }
                for p in self.ordered_passes
            },
        }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def find_project_root(start: Path | str = ".") -> Path:
    """Walk upward from *start* looking for a project.

    Lets you run ``shadertoy render`` from a subdirectory, like git.
    """
    start = Path(start).expanduser().resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if any((candidate / n).is_file() for n in CONFIG_NAMES):
            return candidate
        if _find_first(candidate, _PASS_FILENAMES["image"]):
            return candidate
    raise ProjectError(
        f"No Shadertoy project found in {start} or any parent directory.\n"
        "A project needs an image.glsl (or a shadertoy.toml).\n"
        "Run `shadertoy init` to create one."
    )


def _find_first(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectError(f"{path} is not valid UTF-8: {exc}") from exc


def _infer_kind(source: str) -> str:
    if source == "keyboard":
        return "keyboard"
    if source in BUFFER_NAMES:
        return "buffer"
    if source in BUILTIN_TEXTURES:
        return "builtin"
    return "texture"


def _parse_channel(
    raw: Any, root: Path, pass_name: str, index: int, declared: set[str]
) -> ChannelBinding:
    """Normalise the several accepted channel spellings into a binding."""
    where = f"[{pass_name}] channel{index}"
    if isinstance(raw, str):
        spec: dict[str, Any] = {"source": raw}
    elif isinstance(raw, dict):
        spec = dict(raw)
        # Accept `path`/`texture`/`file` as aliases for `source`.
        for alias in ("path", "texture", "file"):
            if alias in spec and "source" not in spec:
                spec["source"] = spec.pop(alias)
        # `{"type": "keyboard"}` needs no source.
        if "source" not in spec and spec.get("type") == "keyboard":
            spec["source"] = "keyboard"
    else:
        raise ProjectError(
            f"{where} must be a string or an object, got {type(raw).__name__}"
        )

    source = spec.get("source")
    if not isinstance(source, str) or not source:
        raise ProjectError(f"{where} is missing a source")

    kind = spec.get("type")
    if kind is None:
        kind = _infer_kind(source)
    else:
        kind = str(kind).lower()
        if kind not in CHANNEL_TYPES:
            raise ProjectError(
                f"{where}: type must be one of {', '.join(CHANNEL_TYPES)} "
                f"(got {kind!r})"
            )
        inferred = _infer_kind(source)
        # Only complain when the declared type is genuinely impossible; a
        # 'texture' source is a path and cannot be verified by name alone.
        if kind != inferred and not (kind == "texture" and inferred == "texture"):
            if kind == "buffer" and source not in BUFFER_NAMES:
                raise ProjectError(
                    f"{where}: type is \"buffer\" but source {source!r} is not one of "
                    f"{', '.join(BUFFER_NAMES)}"
                )
            if kind == "builtin" and source not in BUILTIN_TEXTURES:
                raise ProjectError(
                    f"{where}: type is \"builtin\" but source {source!r} is not a "
                    f"builtin. Available: {', '.join(BUILTIN_TEXTURES)}"
                )
            if kind == "keyboard" and source != "keyboard":
                raise ProjectError(
                    f"{where}: type is \"keyboard\" but source is {source!r}"
                )
            if kind == "texture" and inferred in ("buffer", "builtin", "keyboard"):
                raise ProjectError(
                    f"{where}: type is \"texture\" but source {source!r} is a reserved "
                    f"{inferred} name. Rename the file or set type to {inferred!r}."
                )

    binding = ChannelBinding(
        source=source,
        kind=kind,
        filter=str(spec.get("filter", "linear")).lower(),
        wrap=str(spec.get("wrap", "repeat")).lower(),
        vflip=bool(spec.get("vflip", True)),
    )
    if binding.filter not in _FILTERS:
        raise ProjectError(
            f"{where}: filter must be one of {', '.join(_FILTERS)} "
            f"(got {binding.filter!r})"
        )
    if binding.wrap not in _WRAPS:
        raise ProjectError(
            f"{where}: wrap must be one of {', '.join(_WRAPS)} (got {binding.wrap!r})"
        )

    if binding.is_buffer:
        if source not in declared:
            raise ProjectError(
                f"{where} reads {source!r}, but this project has no {source}.glsl"
            )
        # Nearest is the safe default for feedback buffers; linear filtering of a
        # buffer you are also writing tends to smear state unintentionally.
        if "filter" not in spec:
            binding.filter = "nearest"
        if "wrap" not in spec:
            binding.wrap = "clamp"
        if "vflip" not in spec:
            binding.vflip = False
    elif binding.is_builtin:
        if "vflip" not in spec:
            binding.vflip = False
    else:
        path = (root / source).resolve()
        if not path.is_file():
            raise ProjectError(
                f"{where}: texture {source!r} not found at {path}.\n"
                f"Expected a file path relative to the project root, a buffer name "
                f"({', '.join(BUFFER_NAMES)}), or a builtin "
                f"({', '.join(BUILTIN_TEXTURES)})."
            )
        binding.path = path
    return binding


def _load_config(path: Path) -> dict[str, Any]:
    """Parse a JSON or TOML config file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProjectError(
                f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: "
                f"{exc.msg}"
            ) from exc
    else:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ProjectError(f"{path}: invalid TOML: {exc}") from exc
    if not isinstance(data, dict):
        raise ProjectError(f"{path}: top level must be an object")
    return data


def load_project(path: Path | str = ".", *, search_parents: bool = True) -> Project:
    """Load the project rooted at (or above) *path*."""
    given = Path(path).expanduser().resolve()
    root = find_project_root(given) if search_parents else given
    if not root.is_dir():
        raise ProjectError(f"{root} is not a directory")

    config: dict[str, Any] = {}
    config_path = _find_first(root, CONFIG_NAMES)
    if config_path is not None:
        config = _load_config(config_path)
        known = {*PASS_NAMES, "defaults", "common", "name", "description"}
        unknown = sorted(set(config) - known)
        if unknown:
            raise ProjectError(
                f"{config_path}: unknown top-level key(s): {', '.join(unknown)}\n"
                f"Expected any of: {', '.join(sorted(known))}"
            )

    # Discover pass files. A pass may also be declared in config via `file`.
    passes: dict[str, PassSpec] = {}
    for name in PASS_NAMES:
        table = config.get(name, {})
        if not isinstance(table, dict):
            raise ProjectError(
                f'{config_path}: "{name}" must be an object, '
                f"got {type(table).__name__}"
            )
        unknown = sorted(set(table) - {"file", "scale", "channels"})
        if unknown:
            raise ProjectError(
                f'{config_path}: "{name}" has unknown key(s): {", ".join(unknown)}\n'
                'Expected any of: channels, file, scale'
            )
        declared_file = table.get("file")
        if declared_file:
            candidate = (root / str(declared_file)).resolve()
            if not candidate.is_file():
                raise ProjectError(
                    f"[{name}] file = {declared_file!r} does not exist at {candidate}"
                )
        else:
            candidate = _find_first(root, _PASS_FILENAMES[name])
            if candidate is None:
                continue
        scale = table.get("scale", 1.0)
        if not isinstance(scale, (int, float)) or not 0 < float(scale) <= 1:
            raise ProjectError(
                f"[{name}] scale must be a number in (0, 1], got {scale!r}"
            )
        passes[name] = PassSpec(
            name=name, path=candidate, source=_read(candidate), scale=float(scale)
        )

    if "image" not in passes:
        tried = ", ".join(_PASS_FILENAMES["image"])
        raise ProjectError(
            f"{root} has no image pass. Expected one of: {tried}\n"
            "Run `shadertoy init` to scaffold a project."
        )

    common_path = _find_first(root, _COMMON_FILENAMES)
    common_file = config.get("common", {}).get("file") if isinstance(
        config.get("common"), dict
    ) else None
    if common_file:
        common_path = (root / str(common_file)).resolve()
        if not common_path.is_file():
            raise ProjectError(f"[common] file = {common_file!r} not found")

    # Resolve channel bindings now so errors surface before touching the GPU.
    declared = set(passes)
    for name, spec in passes.items():
        table = config.get(name, {})
        channels = table.get("channels", {}) if isinstance(table, dict) else {}
        if not isinstance(channels, dict):
            raise ProjectError(
                f'{config_path}: "{name}".channels must be an object keyed by '
                f'channel index, got {type(channels).__name__}'
            )
        valid_keys = {str(i) for i in range(4)} | {f"channel{i}" for i in range(4)}
        unknown = sorted(set(channels) - valid_keys)
        if unknown:
            raise ProjectError(
                f'{config_path}: "{name}".channels has invalid key(s): '
                f'{", ".join(unknown)}\n'
                'Channels are keyed "0".."3" (or "channel0".."channel3").'
            )
        for index in range(4):
            raw = channels.get(str(index), channels.get(f"channel{index}"))
            if raw is None:
                continue
            spec.channels[index] = _parse_channel(raw, root, name, index, declared)

    return Project(
        root=root,
        passes=passes,
        common=_read(common_path) if common_path else None,
        common_path=common_path,
        config_path=config_path,
        config=config,
    )
