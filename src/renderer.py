"""The render engine: pass graph, ping-pong buffers, deterministic time.

Design decisions worth knowing:

**Float targets.** Every pass renders to ``RGBA32F``, not ``RGBA8``. Shadertoy's
buffer passes are float anyway, and keeping the image pass float too means NaN,
Inf and out-of-range values survive to be *detected* rather than being silently
clamped into plausible-looking bytes. Quantisation to 8-bit happens only when
writing a PNG.

**Buffer ordering and feedback.** Passes run A, B, C, D, Image. Each buffer pass
owns two textures and its read/write pair is swapped immediately after it
renders. Consequently a pass reading *another* buffer sees that buffer's
current-frame output, while a pass reading *itself* sees the previous frame --
which is exactly Shadertoy's behaviour and what feedback effects depend on.

**Determinism.** Time is derived from the frame index (``iTime = frame / fps``),
never from a wall clock. When a project has buffer passes, requesting frame N
simulates frames 0..N, because a feedback buffer's contents are a function of
its whole history. With no buffers every frame is a pure function of the
uniforms, so frame N is rendered directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np

from .channels import ChannelTextures
from .context import activate
from .compose import ComposedShader, compose_pass, vertex_source
from .diagnostics import Diagnostic, parse_log
from .inputs import KeyboardState, MouseState
from .project import PassSpec, Project

#: Refuse to simulate more than this many frames without an explicit opt-in.
DEFAULT_MAX_FRAMES = 100_000

#: Fixed date so ``iDate`` cannot make a render irreproducible.
DEFAULT_DATE = (2024.0, 1.0, 1.0, 0.0)


class RenderError(RuntimeError):
    """Raised when rendering cannot proceed."""


class ShaderCompileError(RenderError):
    """A pass failed to compile or link."""

    def __init__(self, diagnostics: list[Diagnostic], composed: ComposedShader | None):
        self.diagnostics = diagnostics
        self.composed = composed
        first = next((d.message for d in diagnostics if d.is_error), "unknown error")
        super().__init__(first)


@dataclass
class RenderSettings:
    """Everything that determines what a render produces."""

    width: int = 640
    height: int = 360
    fps: float = 60.0
    #: Target frame index to capture.
    frame: int = 0
    #: Explicit ``iTime`` override; when None, derived from frame/fps.
    time: float | None = None
    mouse: MouseState = field(default_factory=MouseState)
    keyboard: KeyboardState = field(default_factory=KeyboardState)
    date: tuple[float, float, float, float] = DEFAULT_DATE
    sample_rate: float = 44100.0
    #: None = decide automatically from whether the project has buffers.
    simulate: bool | None = None
    max_frames: int = DEFAULT_MAX_FRAMES

    def time_at(self, frame: int) -> float:
        if self.time is not None:
            # An explicit time pins the captured frame; earlier simulated frames
            # still advance normally so feedback remains coherent.
            if frame == self.frame:
                return self.time
        return frame / self.fps if self.fps else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame": self.frame,
            "time": self.time,
            "mouse": self.mouse.to_dict(),
            "keyboard": self.keyboard.to_dict(),
            "date": list(self.date),
        }


@dataclass
class FrameCapture:
    """One captured frame: float pixel data for each pass."""

    frame: int
    time: float
    #: Pass name -> ``(h, w, 4)`` float32 array in GL order (row 0 = bottom).
    images: dict[str, np.ndarray]
    #: Wall-clock GPU time for this frame, in milliseconds.
    duration_ms: float = 0.0


class _Target:
    """A render target; buffer passes get two for ping-ponging."""

    def __init__(self, ctx: Any, size: tuple[int, int], double: bool) -> None:
        self.ctx = ctx
        self.size = size
        #: Set when some pass samples this buffer with mipmap filtering, which
        #: requires regenerating the pyramid after every write.
        self.needs_mipmaps = False
        self._textures = [self._make(ctx, size)]
        self._fbos = [ctx.framebuffer(color_attachments=[self._textures[0]])]
        if double:
            self._textures.append(self._make(ctx, size))
            self._fbos.append(ctx.framebuffer(color_attachments=[self._textures[1]]))
        self._read = 0
        # Start from a defined state; an unwritten buffer would otherwise expose
        # driver-dependent garbage on frame 0.
        for fbo in self._fbos:
            fbo.clear(0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def _make(ctx: Any, size: tuple[int, int]) -> Any:
        texture = ctx.texture(size, 4, dtype="f4")
        texture.filter = (ctx.NEAREST, ctx.NEAREST)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    @property
    def double(self) -> bool:
        return len(self._textures) > 1

    @property
    def read_texture(self) -> Any:
        return self._textures[self._read]

    @property
    def write_fbo(self) -> Any:
        return self._fbos[-1 - self._read] if self.double else self._fbos[0]

    @property
    def write_texture(self) -> Any:
        return self._textures[-1 - self._read] if self.double else self._textures[0]

    def swap(self) -> None:
        if self.double:
            self._read = 1 - self._read
        if self.needs_mipmaps:
            # Must happen after the swap: read_texture is the one just written,
            # and a stale pyramid would silently serve last frame's lower levels.
            self.read_texture.build_mipmaps()

    def read_array(self) -> np.ndarray:
        fbo = self._fbos[self._read] if self.double else self._fbos[0]
        raw = fbo.read(components=4, dtype="f4")
        width, height = self.size
        return np.frombuffer(raw, dtype=np.float32).reshape(height, width, 4).copy()

    def release(self) -> None:
        for item in (*self._fbos, *self._textures):
            try:
                item.release()
            except Exception:  # pragma: no cover
                pass


class _Pass:
    """A compiled pass with its program and target."""

    def __init__(self, spec: PassSpec, composed: ComposedShader, program: Any) -> None:
        self.spec = spec
        self.composed = composed
        self.program = program
        self.target: _Target | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    def release(self) -> None:
        if self.target is not None:
            self.target.release()
        try:
            self.program.release()
        except Exception:  # pragma: no cover
            pass


def _set_uniform(program: Any, name: str, value: Any) -> None:
    """Assign a uniform, tolerating ones the compiler optimised away."""
    uniform = program.get(name, None)
    if uniform is None:
        return
    try:
        uniform.value = value
    except Exception:
        # Arrays and some vector types must go through write().
        array = np.asarray(value, dtype="f4")
        # Drivers disagree about how many elements of an array uniform they
        # expose: NVIDIA reports the full declared length, while Mesa trims it
        # to the elements the shader actually indexes. Writing more bytes than
        # the program expects fails with "invalid uniform size", so clamp to
        # whatever this program says it has. A shader using only
        # iChannelResolution[0] is extremely common, which makes this the
        # difference between working and crashing on Mesa.
        length = getattr(uniform, "array_length", None) or 1
        if array.ndim >= 1 and array.shape[0] > length:
            array = array[:length]
        uniform.write(np.ascontiguousarray(array).tobytes())


class Renderer:
    """Compiles and runs a project's pass graph."""

    def __init__(self, project: Project, ctx: Any, settings: RenderSettings) -> None:
        self.project = project
        self.ctx = ctx
        self.settings = settings
        self.passes: dict[str, _Pass] = {}
        self.channels = ChannelTextures(ctx)
        self._quad: Any | None = None
        self._vaos: dict[str, Any] = {}
        self._compiled = False

    # -- compilation -----------------------------------------------------

    def compile(self, *, collect_all: bool = False) -> list[Diagnostic]:
        """Compile every pass.

        With *collect_all*, compilation continues past a failing pass so a single
        run reports every broken pass instead of only the first.
        """
        import moderngl

        # GL calls target whichever context is current; a process may hold more
        # than one, so claim ours before touching the driver.
        activate(self.ctx)
        version = int(self.project.default("glsl_version", 330))
        vertex = vertex_source(version)
        diagnostics: list[Diagnostic] = []
        failures: list[tuple[list[Diagnostic], ComposedShader]] = []

        for spec in self.project.ordered_passes:
            composed = compose_pass(self.project, spec, version=version)
            try:
                program = self.ctx.program(
                    vertex_shader=vertex, fragment_shader=composed.source
                )
            except moderngl.Error as exc:
                found = parse_log(str(exc), composed, spec.name)
                if not any(d.is_error for d in found):
                    found.append(
                        Diagnostic(
                            severity="error",
                            message=str(exc).strip(),
                            pass_name=spec.name,
                        )
                    )
                diagnostics.extend(found)
                failures.append((found, composed))
                if not collect_all:
                    raise ShaderCompileError(found, composed) from None
                continue
            diagnostics.extend(parse_log("", composed, spec.name))
            self.passes[spec.name] = _Pass(spec, composed, program)

        if failures and not collect_all:  # pragma: no cover - handled above
            raise ShaderCompileError(*failures[0])
        self._compiled = not failures
        return diagnostics

    # -- setup -----------------------------------------------------------

    def _ensure_targets(self) -> None:
        width, height = self.settings.width, self.settings.height
        for name, rp in self.passes.items():
            if rp.target is not None:
                continue
            scale = rp.spec.scale
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            rp.target = _Target(self.ctx, size, double=not rp.spec.is_image)

        # A buffer needs mipmaps if *any* pass samples it with mipmap filtering.
        for rp in self.passes.values():
            for binding in rp.spec.channels.values():
                if not binding.is_buffer or binding.filter != "mipmap":
                    continue
                source = self.passes.get(binding.source)
                if source is not None and source.target is not None:
                    source.target.needs_mipmaps = True

    def _ensure_geometry(self) -> None:
        if self._quad is None:
            # Full-screen triangle strip; fragCoord comes from gl_FragCoord.
            data = np.array(
                [-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"
            ).tobytes()
            self._quad = self.ctx.buffer(data)
        for name, rp in self.passes.items():
            if name not in self._vaos:
                self._vaos[name] = self.ctx.vertex_array(
                    rp.program, [(self._quad, "2f", "_st_position")]
                )

    # -- rendering -------------------------------------------------------

    def _bind_channels(self, rp: _Pass, frame: int) -> list[tuple[int, int, int]]:
        """Bind this pass's channels; returns per-channel (w, h, depth)."""
        resolutions = [(0, 0, 0)] * 4
        for index in range(4):
            binding = rp.spec.channels.get(index)
            if binding is None:
                continue
            if binding.is_buffer:
                source = self.passes.get(binding.source)
                if source is None or source.target is None:
                    raise RenderError(
                        f"{rp.spec.label} channel{index} reads {binding.source}, "
                        "which failed to compile"
                    )
                texture = source.target.read_texture
                # Buffer sampling honours the binding's filter/wrap settings.
                if binding.filter == "nearest":
                    texture.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
                elif binding.filter == "mipmap":
                    texture.filter = (
                        self.ctx.LINEAR_MIPMAP_LINEAR,
                        self.ctx.LINEAR,
                    )
                else:
                    texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
                repeat = binding.wrap == "repeat"
                texture.repeat_x = repeat
                texture.repeat_y = repeat
                size = source.target.size
            elif binding.is_keyboard:
                texture = self.channels.keyboard(self.settings.keyboard, frame)
                size = texture.size
            else:
                texture = self.channels.get(binding)
                size = texture.size
            texture.use(index)
            _set_uniform(rp.program, f"iChannel{index}", index)
            resolutions[index] = (size[0], size[1], 1)
        return resolutions

    def _set_frame_uniforms(
        self, rp: _Pass, frame: int, now: float, resolutions: list[tuple[int, int, int]]
    ) -> None:
        target = rp.target
        assert target is not None
        width, height = target.size
        program = rp.program
        _set_uniform(program, "iResolution", (float(width), float(height), 1.0))
        _set_uniform(program, "iTime", now)
        _set_uniform(program, "iTimeDelta", 1.0 / self.settings.fps if self.settings.fps else 0.0)
        _set_uniform(program, "iFrameRate", float(self.settings.fps))
        _set_uniform(program, "iFrame", int(frame))
        _set_uniform(program, "iSampleRate", float(self.settings.sample_rate))
        _set_uniform(program, "iDate", tuple(float(v) for v in self.settings.date))
        _set_uniform(
            program,
            "iMouse",
            self.settings.mouse.as_vec4(width, height, frame),
        )
        _set_uniform(program, "iChannelTime", [now] * 4)
        _set_uniform(
            program,
            "iChannelResolution",
            [[float(c) for c in res] for res in resolutions],
        )

    def run(self, capture: Iterable[int] | None = None) -> Iterator[FrameCapture]:
        """Simulate frames, yielding a capture for each requested frame index."""
        if not self.passes:
            raise RenderError("no passes compiled; call compile() first")

        wanted = sorted(set(capture if capture is not None else [self.settings.frame]))
        if not wanted:
            return
        if any(f < 0 for f in wanted):
            raise RenderError("frame indices must be >= 0")

        last = max(wanted)
        simulate = self.settings.simulate
        if simulate is None:
            # Feedback buffers depend on their entire history; a plain shader
            # does not, so it can jump straight to the requested frame.
            simulate = bool(self.project.buffer_passes)
        if last + 1 > self.settings.max_frames:
            raise RenderError(
                f"frame {last} requires simulating {last + 1} frames, above the "
                f"limit of {self.settings.max_frames}. Raise --max-frames to allow it."
            )

        activate(self.ctx)
        self._ensure_targets()
        self._ensure_geometry()

        ctx = self.ctx
        ctx.disable(ctx.DEPTH_TEST)
        ctx.disable(ctx.BLEND)
        ctx.disable(ctx.CULL_FACE)

        timeline = range(0, last + 1) if simulate else wanted
        wanted_set = set(wanted)

        for frame in timeline:
            # Re-claim each frame: control returned to the caller at the last
            # yield, and it may have used a different context meanwhile.
            activate(self.ctx)
            now = self.settings.time_at(frame)
            start = time.perf_counter()
            for rp in self.project.ordered_passes:
                compiled = self.passes.get(rp.name)
                if compiled is None or compiled.target is None:
                    continue
                resolutions = self._bind_channels(compiled, frame)
                self._set_frame_uniforms(compiled, frame, now, resolutions)
                compiled.target.write_fbo.use()
                self._vaos[rp.name].render(mode=self.ctx.TRIANGLE_STRIP)
                # Swap now so later passes in this frame observe fresh output.
                compiled.target.swap()
            ctx.finish()
            duration_ms = (time.perf_counter() - start) * 1000.0

            if frame in wanted_set:
                images = {
                    name: rp.target.read_array()
                    for name, rp in self.passes.items()
                    if rp.target is not None
                }
                yield FrameCapture(
                    frame=frame, time=now, images=images, duration_ms=duration_ms
                )

    def render_frame(self, frame: int | None = None) -> FrameCapture:
        """Render and return a single frame."""
        target = self.settings.frame if frame is None else frame
        for capture in self.run([target]):
            return capture
        raise RenderError(f"frame {target} produced no output")

    def release(self) -> None:
        # Freeing GL objects also requires the owning context to be current.
        try:
            activate(self.ctx)
        except Exception:  # pragma: no cover - teardown is best effort
            pass
        for vao in self._vaos.values():
            try:
                vao.release()
            except Exception:  # pragma: no cover
                pass
        self._vaos.clear()
        if self._quad is not None:
            try:
                self._quad.release()
            except Exception:  # pragma: no cover
                pass
            self._quad = None
        for rp in self.passes.values():
            rp.release()
        self.passes.clear()
        self.channels.release()
