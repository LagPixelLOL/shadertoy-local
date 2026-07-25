# shadertoy-local

Headless local Shadertoy runtime and shader test harness, built for automated
and agentic development.

It treats a directory as a Shadertoy project, renders it on the GPU without a
display server, and reports results in a form a program can act on: errors
mapped to real source lines, pixel probes, frame statistics, and golden-image
regression tests. It is a **dev/test tool first** and a renderer second.

```bash
pip install -e .

shadertoy init -t feedback     # scaffold a project
shadertoy check                # compile, report errors with file:line
shadertoy render --frame 60    # render a PNG
shadertoy stats --assert-finite --assert-not-uniform
```

## Why not just use shadertoy.com

Because a browser tab cannot be asserted against. This tool exists to answer
questions mechanically:

- Did the shader compile, and **which line** of *my* file is wrong?
- Is the output actually correct, or merely non-crashing?
- Did that refactor change any pixels?
- Does it still work when the mouse is held at (320, 180) with `W` down?

## Project layout

A project is just a folder. Only `image.glsl` is required.

```
myshader/
  shadertoy.json      # optional: channel wiring and defaults
  common.glsl         # optional: prepended to every pass
  image.glsl          # required: the final pass
  buffer_a.glsl       # optional: Buffer A .. Buffer D
  textures/wood.png
  golden/             # reference images for `shadertoy test`
  out/                # render output (gitignored)
```

**Passes are identified by filename, never by config.** `image.glsl` is the
image pass, `buffer_a.glsl` is Buffer A, `common.glsl` is shared code. The
config file only describes *wiring*, so a file's role is always obvious from its
name. Passes execute in Shadertoy's order: Buffer A, B, C, D, then Image.

Each pass gets the standard Shadertoy uniforms (`iResolution`, `iTime`,
`iTimeDelta`, `iFrameRate`, `iFrame`, `iChannelTime`, `iChannelResolution`,
`iMouse`, `iDate`, `iSampleRate`, `iChannel0..3`) plus `texture2D`,
`textureCube` and `iGlobalTime` aliases so shaders copied from the site
generally work unmodified. `#include "file.glsl"` is also supported, which
Shadertoy lacks.

## Configuration

`shadertoy.json` (or `shadertoy.toml`, same schema):

```json
{
  "defaults": { "width": 640, "height": 360, "fps": 60 },
  "buffer_a": {
    "scale": 0.5,
    "channels": {
      "0": { "type": "buffer", "source": "buffer_a" }
    }
  },
  "image": {
    "channels": {
      "0": { "type": "buffer",  "source": "buffer_a" },
      "1": { "type": "texture", "source": "textures/wood.png",
             "filter": "linear", "wrap": "repeat", "vflip": true },
      "2": { "type": "builtin", "source": "noise" },
      "3": { "type": "keyboard" }
    }
  }
}
```

| Channel type | `source` | Notes |
|---|---|---|
| `buffer` | `buffer_a`..`buffer_d` | Defaults to `nearest`/`clamp`, which is safer for feedback |
| `texture` | path relative to project root | Any format Pillow reads |
| `builtin` | see below | Procedural, needs no asset files |
| `keyboard` | *(none needed)* | 256x3 key state texture |

Builtin textures are deterministic (fixed seed), so golden tests stay stable:
`noise`, `rgba-noise`, `gray-noise`, `blue-noise`, `checker`, `uv`, `gradient`,
`white`, `black`.

`"0": "buffer_a"` is shorthand for the full object; `type` is inferred from
`source` and validated when given, so a typo becomes a clear error.

## Commands

| Command | Purpose |
|---|---|
| `info` | GPU, EGL devices, active context, project layout |
| `init` | Scaffold a project (`basic`, `common`, `feedback`, `input`) |
| `check` | Compile every pass, report errors. No rendering |
| `render` | Render frames to PNG |
| `probe` | Read exact pixel values, optionally asserting them |
| `stats` | Frame statistics with optional assertions |
| `test` | Compare renders against `golden/` |
| `bless` | Write or update `golden/` |

Exit codes: **0** ok, **1** shader or assertion failure, **2** usage/project
error, **3** no usable GPU.

Every command accepts `--json` and writes a single object to **stdout**, with
all prose on **stderr** — so `--json` output is always parseable, while errors
still explain themselves.

## Determinism

Time comes from the frame index (`iTime = frame / fps`), never a wall clock, and
`iDate` is fixed by default. Identical arguments therefore produce
byte-identical pixels (there is a test asserting exactly that).

Because a feedback buffer's contents depend on its whole history, requesting
frame *N* simulates frames 0..*N* when the project has buffer passes. Without
buffers, every frame is a pure function of the uniforms, so frame *N* renders
directly. Override with `--simulate` / `--no-simulate`.

## Simulated input

No window, no real devices — input is supplied as arguments and is fully
reproducible.

```bash
shadertoy render --mouse 320,180                  # held at a position
shadertoy render --mouse 0.5,0.5 --mouse-norm     # fractions of resolution
shadertoy render --mouse 320,180 --mouse-button click
shadertoy render --key w,a --key-press space --key-toggle g
```

`iMouse` follows Shadertoy's sign encoding: `xy` is the cursor, `abs(zw)` is
where the click began, `z > 0` means held, `w > 0` means this is the press
frame.

The keyboard is a 256x3 texture indexed by JavaScript key code:

```glsl
float held    = texelFetch(iChannel0, ivec2(87, 0), 0).x;  // row 0: held
float pressed = texelFetch(iChannel0, ivec2(87, 1), 0).x;  // row 1: pressed
float toggled = texelFetch(iChannel0, ivec2(87, 2), 0).x;  // row 2: toggle
```

Keys accept names or codes: `w`, `space`, `left`, `f1`, `numpad0`, `27`.

## Verifying output without looking at it

Passes render to `RGBA32F`, not `RGBA8`. That is deliberate: NaN, Inf and
out-of-range values survive to be *detected* instead of being silently clamped
into plausible-looking bytes. Quantisation happens only when writing a PNG.

```bash
# Exact pixel assertions
shadertoy probe --at 10,50=1,0,0 --at 90,50=0,0,1
shadertoy probe --at n:0.5,0.5          # normalised coordinates

# Fail the build on a broken frame
shadertoy stats --assert-finite --assert-not-black --assert-not-uniform \
                --min-unique-colors 64
```

`stats` also reports per-channel ranges, luma, unique colour count, and the
fraction of pixels clipped or out of range — useful for spotting a shader that
only looks fine because the display clamps it.

## Regression testing

```bash
shadertoy bless --frames 0,30,60      # record references
shadertoy test  --frames 0,30,60      # compare (exit 1 on drift)
```

References are 8-bit PNGs in `golden/`: small, reviewable in a pull request, and
immune to last-bit floating point differences between drivers. Comparison uses
**both** a max and a mean tolerance (`--max-diff`, `--mean-diff`) because max
alone is brittle and mean alone hides small bright artefacts. On failure the
rendered frame and an amplified diff are written for inspection.

## Error reporting

The generated uniform prelude means a driver error on "line 37" is nowhere near
line 37 of your file. Every position is mapped back through an explicit per-line
origin table, which also works across the `common.glsl` boundary:

```
$ shadertoy check
common.glsl:2: error [C1503]: undefined variable "bogus_r"
     1 | // shared helpers
  >  2 | float sdCircle(vec2 p, float r) { return length(p) - bogus_r; }
     3 |

image.glsl:5: error [C1503]: undefined variable "undefined_variable"
     4 |     vec3 col = vec3(smoothstep(0.0, 0.01, d));
  >  5 |     col *= undefined_variable;
     6 |     fragColor = vec4(col, 1.0);
```

NVIDIA, Mesa and AMD log formats are all parsed, and unrecognised lines are
surfaced rather than swallowed. Positions embedded in a message body (such as
NVIDIA's `conflicts with previous declaration at 0(5)`) are remapped too.

## Common-tab portability

shadertoy-local concatenates `common.glsl` into every pass *after* the uniform
prelude, so Common code may reference `iTime` and friends freely. **The real site
does not work that way.** It validates the Common tab standalone against a
minimal header in which only `iDate` and `iSampleRate` exist, so identical code
shows `undeclared identifier` in the site's editor while still rendering
correctly.

That noise is cosmetic but not harmless: ~30 lines of spurious uniform errors
will camouflage a genuine typo in Common. `shadertoy check` therefore warns about
it **by default** (as warnings — the exit code stays 0):

```
$ shadertoy check
common.glsl:2:41: warning [ST-COMMON]: iTime is not visible in shadertoy.com's Common tab
ok: 1 pass(es) compiled [Image], 1 portability warning(s)
```

Disable with `--no-portable-common` if the shader will never go back to the site.

Comments are ignored, and so are `#define` bodies — an unexpanded macro is never
compiled, so the site does not flag it either. That exemption is what makes the
recommended fix ergonomic: capture uniforms into a struct once per pass and take
it as a parameter in Common. See `examples/06-portable-common`:

```glsl
// common.glsl -- validates clean
struct ST { vec3 res; float time; float dt; int frame; vec4 mouse; };
#define ST_CAPTURE ST(iResolution, iTime, iTimeDelta, iFrame, iMouse)
float aspect(ST u) { return u.res.x / u.res.y; }

// image.glsl -- uniforms read where they legally exist
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ST u = ST_CAPTURE;
    ...
}
```

Do **not** try to fix it by declaring `uniform float iTime;` in Common: that
collides with the pass's own declaration. shadertoy-local reproduces the failure
faithfully, and tells you where the conflict is.

## Requirements

Python 3.11+, `moderngl`, `numpy`, `pillow`, and a GPU reachable through EGL.

Rendering is fully headless via `EGL_EXT_platform_device`, so no X server,
Wayland session or `/dev/dri` node is needed. Hardware devices are preferred
automatically — a software rasterizer is refused unless you pass
`--allow-software`, since silently falling back to llvmpipe turns a 1 ms render
into a 1 s one.

Check the environment with `shadertoy info`. If EGL setup is incomplete the
error explains the fix; on Debian/Ubuntu with NVIDIA that is usually:

```bash
apt-get install -y --no-install-recommends libegl1
# and a GLVND vendor config so EGL can see the NVIDIA driver:
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

## Examples

Each directory in `examples/` is a runnable project with its own
`shadertoy.json`:

| Example | Demonstrates |
|---|---|
| `01-plasma` | Minimal single pass, no inputs |
| `02-raymarch` | `common.glsl` for shared SDF helpers |
| `03-feedback-trail` | Buffer feedback, and self-read vs cross-read timing |
| `04-textured` | Builtin textures across filter/wrap modes |
| `05-interactive` | Keyboard rows and `iMouse` |
| `06-portable-common` | Uniform-struct protocol for a lint-clean Common tab |

```bash
shadertoy render -C examples/03-feedback-trail --frame 120
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # everything
pytest -m "not gpu"       # logic only, no GPU needed
```

## License

MIT
