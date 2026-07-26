# Shadertoy Local

Warning: This project is basically entirely AI vibe shidded LOL.

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
| `buffer` | `buffer_a`..`buffer_d` | Defaults to `linear`/`clamp` — matching shadertoy.com. Settings are per *buffer*, see below |
| `texture` | path relative to project root | Any format Pillow reads |
| `builtin` | see below | Procedural, needs no asset files |
| `keyboard` | *(none needed)* | 256x3 key state texture. Filter `linear` (default) or `nearest` only, wrap is always `clamp` — matching the site's dialog |

Builtin textures are procedurally generated and deterministic (fixed seed), so
golden tests stay stable. They fall into two groups, and the distinction matters
for portability:

**Mirror a stock shadertoy.com input** — same role and same dimensions, so a
ported shader samples the right kind of data at the right resolution. The pixel
values are *not* identical, since those assets cannot be redistributed:

| Builtin | Size | Corresponds to |
|---|---|---|
| `rgba-noise-small` | 64 | RGBA Noise Small |
| `rgba-noise-medium` (= `noise`, `rgba-noise`) | 256 | RGBA Noise Medium |
| `gray-noise-small` | 64 | Gray Noise Small |
| `gray-noise-medium` (= `gray-noise`) | 256 | Gray Noise Medium |
| `blue-noise` | 1024 | Blue Noise (approximated by high-pass filtering) |
| `bayer` | 16 | Bayer — **exact**, being defined by recurrence rather than authored |

**Local-only debug aids**, with no counterpart on the site: `checker`, `uv`,
`gradient`, `white`, `black`. Useful for checking orientation, wrapping and
filtering, but a project using them cannot be reproduced there by binding a stock
input.

Sizes are assumed from the site's assets; override with `"size": N` on any
builtin if one is wrong.

`"0": "buffer_a"` is shorthand for the full object; `type` is inferred from
`source` and validated when given.

### Sampler settings belong to the input, not the channel

On shadertoy.com `filter` and `wrap` are properties of the **input** — the
buffer, the texture asset, or the keyboard — not of the channel reading it:
change them on one reference and every reference changes, because GL stores
sampler state on the texture object and the site has one texture object per
input. Asking for two different settings for one input is therefore
inexpressible on the real site, and is rejected here rather than resolved in
favour of whichever binding is applied last:

```
error: [image] channel1 reads buffer_a with filter=nearest, wrap=clamp, but
[image] channel0 reads it with filter=mipmap, wrap=repeat.
A buffer's sampler settings belong to the buffer, not to the channel: on
shadertoy.com changing them on one reference changes every reference.
```

The identity is the underlying object, however it is spelled: `noise` and
`rgba-noise-medium` are one asset, a builtin at its default size and the same
builtin with that size written out are one texture, and so are two spellings
of one file path. `vflip` participates for image-backed inputs, since the
flip is baked into the pixel upload; it is rejected outright on buffer and
keyboard channels, where there is nothing to flip.

The constraint is per input, so two different buffers — or two different
sizes of one builtin, which really are two textures — may use different
settings.

The keyboard is further restricted to what the site's dialog offers: filter
`nearest` or `linear` (no mipmap) and wrap `clamp`, always. Anything else is
rejected at `shadertoy check` time.

### Validation is strict

Config is checked exhaustively rather than read leniently, because a mistyped key
that is merely ignored is the worst available outcome: the shader renders, at the
wrong size or with the wrong sampler, and nothing says so. Every key is checked
against an allow-list and every value against its type and range, so mistakes fail
at `shadertoy check` rather than surfacing as a puzzling image:

```
$ shadertoy check
error: shadertoy.json: defaults: unknown key(s): 'widht' (did you mean 'width'?)
Allowed keys: fps, glsl_version, height, width
```

Specifically rejected: unknown keys at every level (with a "did you mean"
suggestion), non-integer or out-of-range `width`/`height`/`fps`/`glsl_version`,
`scale` outside `(0, 1]`, anything but a real `true`/`false` for `vflip` (so
`"yes"` and `1` are errors), unknown `filter`/`wrap`/`type` values, `size` on a
non-builtin channel, giving one channel twice as both `"0"` and `"channel0"`, and
synonyms for `source` — it is spelled `source`, not `path`, `texture` or `file`.
TOML is validated identically, since the schema is shared.

## Commands

| Command | Purpose |
|---|---|
| `info` | GPU, EGL devices, verification shader per device, and a shadertoy.com porting guide for the project |
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

### Warm-up and multiple frames

A feedback buffer's contents depend on its whole history, so a project with buffer
passes renders every frame from 0 up to the one requested. Without buffers each
frame is a pure function of the uniforms and renders directly.

`--precharge` overrides that with a single knob:

| Value | Effect |
|---|---|
| *(unset)* | `all` when the project has buffer passes, otherwise `0` |
| `all` | Warm up from frame 0 |
| `N` | Render only N warm-up frames before the first capture |
| `0` | No warm-up; render the captured frames only |

`--precharge 50 --frame 500` is the useful case: enough history for the
accumulator to settle, without paying for 500 frames.

Capture several frames with `--count` and `--every`, starting at `--frame`:

```bash
shadertoy render --count 10 --every 5              # frames 0, 5, ... 45
shadertoy render --frame 100 --count 3 --every 10  # frames 100, 110, 120
shadertoy render --count 8 --every 4 --precharge 20
```

**Frames between captures are really rendered.** `--count 3 --every 20` draws
frames 0..40 and saves three of them; it does not draw only three. Skipping the
gaps would corrupt accumulation, since each capture would then hold the history of
only the frames actually drawn. With an accumulating buffer:

```
captured frame   0  ->  accumulator =  1
captured frame  20  ->  accumulator = 21     # not 2
captured frame  40  ->  accumulator = 41     # not 3
```

`--precharge` governs only the run-up *before the first* capture, never the gaps
between captures — so `--precharge 0 --count 3 --every 20` still renders frames
0..40, it just starts the accumulator cold at the first one.

The one exception is a project with **no buffer passes**: nothing accumulates,
each frame is a pure function of its uniforms, so only the captured frames are
drawn. `--every 200` then costs no more than `--every 20`. There is a test
asserting the skipped and unskipped results are byte-identical, which is what
makes the optimisation safe.

Because of that, `render` reports timing **per frame** and counts the frames you
never see, rather than summing the captures alone:

```
$ shadertoy render --frame 1000 --count 10 --every 100
rendered 10 frame(s) at 640x360 on NVIDIA RTX PRO 6000 Blackwell ...
  gpu time 0.699 ms/frame (min 0.675, max 0.735) x 10 = 6.995 ms
  plus 1891 uncaptured frame(s) 1335.699 ms; 1342.694 ms for all 1901 rendered
```

The per-frame figure is the one that says something about the shader; a total on
its own only restates `--count`. The second line appears when warm-up or
skipped-over frames were drawn, so the reported cost cannot quietly disagree with
the wall clock by two orders of magnitude. `--json` carries the same numbers under
`timing`.

## Simulated input

No window and no real devices: input is a **single timeline of operations**
covering pointer and keyboard together, so a drag, a click-and-release, or a key
pressed at one specific frame is expressible and reproducible.

```bash
shadertoy render --input input.json --frame 80
shadertoy render --input '[{"frame":0,"op":"mouse_down","pos":[320,180]}]'
cat ops.json | shadertoy render --input -
shadertoy render --help-input        # full operation reference
```

Each operation is scheduled by **`frame` or `time`** — frames when reasoning about
simulation steps, seconds when matching what the shader does with `iTime` (times
are converted with `--fps`, rounding to nearest):

```json
[
  {"frame": 0,   "op": "mouse_down", "pos": [320, 180]},
  {"frame": 30,  "op": "mouse_move", "pos": [500, 180]},
  {"time":  1.0, "op": "key_down",   "keys": ["w", "shift"]},
  {"frame": 75,  "op": "key_tap",    "keys": ["space"]},
  {"frame": 90,  "op": "mouse_up"},
  {"frame": 95,  "op": "key_up",     "keys": ["w", "shift"]}
]
```

| Operation | Effect |
|---|---|
| `mouse_down` | Press the button; optional `pos`. Sets the click anchor |
| `mouse_up` | Release; optional `pos` |
| `mouse_move` | Move the cursor; requires `pos` |
| `key_down` | Hold keys until a matching `key_up` |
| `key_up` | Release keys |
| `key_tap` | Hold for exactly that one frame |
| `key_toggle` | Flip the toggle row |
| `key_untoggle` | Clear the toggle row |

`pos` is in pixels, or fractions of the resolution with `"normalized": true`.
`keys` accepts names or numeric JavaScript key codes: `w`, `space`, `left`, `f1`,
`numpad0`, `27`.

**The pointer cannot leave the canvas.** On shadertoy.com the listeners are on the
canvas element and there is no pointer capture, so an event outside it never
fires — `iMouse` simply cannot hold an off-canvas value there. A position outside
`0..width` / `0..height` (or `0..1` when normalized) is therefore rejected rather
than rendered, since a shader tuned against one could not be reproduced on the
site:

```
$ shadertoy render --input '[{"frame": 0, "op": "mouse_down", "pos": [69420, -69]}]'
error: input[0]: pos x=69420 is outside the 640x360 canvas (0..640). The pointer
cannot leave the canvas on shadertoy.com, so iMouse can never hold this value.
```

Negative values are doubly wrong: `iMouse.zw` encodes the button state in its
*signs*, so a negative press position is not off-screen input, it is a corrupted
one. A single timeline must also pick one unit — pixels or normalized — because
the cursor and the click anchor share one scale.

Operations **need not be written in temporal order** — grouping a keypress next to
the drag it accompanies is often clearer. The list is sorted on construction, and
sorting is *stable*, so operations sharing a frame apply in written order
(`mouse_down` then `mouse_move` anchors the click at the press position; the
reverse anchors it at the moved one).

In the shader:

```glsl
iMouse.xy                              // cursor in pixels
iMouse.z > 0.0                         // button held
iMouse.w > 0.0                         // this is the frame of the press
abs(iMouse.zw)                         // where the press began
texelFetch(iChannel0, ivec2(87,0),0).x // key held      (row 0)
texelFetch(iChannel0, ivec2(87,1),0).x // pressed now   (row 1)
texelFetch(iChannel0, ivec2(87,2),0).x // toggle        (row 2)
```

`examples/05-interactive/input.json` is a working timeline; render it at various
frames to watch the state evolve.

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
shadertoy bless --count 3 --every 30      # record references
shadertoy test  --count 3 --every 30      # compare (exit 1 on drift)
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

## shadertoy.com portability

`shadertoy check` also reports where a project would behave differently on the
real site. Severity tracks consequence:

| Code | Severity | Because |
|---|---|---|
| `ST-COMMON` | warning | The shader still compiles and renders on the site; only the editor's standalone validation complains |
| `ST-TERNARY` | **error** | The shader does not compile on the site at all |
| `ST-RESERVED` | **error** | The shader does not compile on the site at all |

All are on by default; `--no-portability` skips them.

### ST-RESERVED: identifiers reserved in GLSL ES

GLSL ES reserves a set of words that desktop drivers accept as plain
identifiers without a murmur — `float active;` compiles on NVIDIA GL 4.6 and
stops the same shader dead on shadertoy.com with `'active' : Illegal use of
reserved word`. `filter`, `input`, `output`, `half`, `sample` and some forty
others are in the same set, and several are natural variable names, which is
exactly how this was found: a shader that passed every local check refused to
compile the moment it was pasted into the site.

```
$ shadertoy check
image.glsl:2:11: error [ST-RESERVED]: 'active' is reserved in GLSL ES and does not compile on shadertoy.com; rename it
failed: 1 error(s), 0 warning(s)
```

Comments are ignored; `#define` bodies are **not** exempt (unlike ST-COMMON),
because a reserved word in a macro fails on the site wherever the macro is
expanded — and the site's own error message will point at the expansion, not
the definition. Real keywords of both dialects (`if`, `uniform`, ...) are not
in the list, since the local compile already rejects them; this is the silent
set.

### ST-TERNARY: `?:` on struct types

Desktop GLSL permits the ternary operator on structs and arrays, so this compiles
here — but shadertoy.com runs WebGL, where `?:` on a struct fails to compile
(notably through ANGLE). A clean local run would therefore be actively
misleading, so it is reported as an **error** and fails the command:

```
$ shadertoy check
common.glsl:5: error [ST-TERNARY]: ?: yields struct 'Ray', which does not compile on shadertoy.com; use if/else
failed: 1 error(s), 0 warning(s)
```

The fix is mechanical and never slower:

```glsl
Ray r = flag ? a : b;              // rejected on shadertoy.com
Ray r = b; if (flag) { r = a; }    // portable
```

Detection is heuristic — it flags a ternary whose result is a struct, via a
struct-typed assignment target, a constructor in a branch, or a bare struct
variable as a branch. `flag ? a.o : b.o` evaluates to a `vec3` and is left alone.

### ST-COMMON: Common-tab uniform visibility

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

Disable with `--no-portability` if the shader will never go back to the site.

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

### Verifying the environment

`shadertoy info` does not merely list devices — it compiles and runs a small
shader on **each** one and checks the output pixel by pixel:

```
$ shadertoy info
EGL devices (with shader runtime check):
  [0] NVIDIA RTX PRO 6000 Blackwell Server Edition/PCIe/SSE2  (hardware)
        runtime: ok    4.6.0 NVIDIA 580.95.05  shader 3.28 ms  (context 138 ms)
  [1] llvmpipe (LLVM 21.1.8, 256 bits)  (software)
        runtime: ok    4.5 (Core Profile) Mesa 26.0.3  shader 15.69 ms  (context 57 ms)
```

Enumerating a device proves nothing: it may refuse to bind, bind but fail to
compile, or compile but render incorrectly. The check therefore exercises exactly
what a real pass depends on — the vertex path, `gl_FragCoord`, scalar, vector and
**array** uniforms, an `RGBA32F` target holding values above 1.0, and readback —
then compares every pixel against values computed on the CPU. A driver that
renders *something* but renders it wrongly fails rather than passes.

Failures report the stage reached, so the cause is narrowed immediately:

```
        runtime: FAILED at compile: GLSL Compiler failed
        runtime: FAILED at verify: rendered image differs from expected by 0.01875
```

Context creation time is reported separately from shader time, because the former
is dominated by one-time driver initialisation and would otherwise make a fast GPU
look slower than a CPU rasterizer. `info` exits 3 if no device can run a shader,
and `--no-runtime-check` skips it for a quick listing.

If EGL setup is incomplete the error explains the fix; on Debian/Ubuntu with
NVIDIA that is usually:

```bash
apt-get install -y libegl1
# and a GLVND vendor config so EGL can see the NVIDIA driver:
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

## Porting back to shadertoy.com

`shadertoy info` reports what to actually do on the site: which tabs to create,
what to select in each `iChannel` slot, the sampler settings, and — most usefully —
which parts of the project **cannot** be reproduced there.

```
-- porting to shadertoy.com ---------------------------------------------
  Tabs to create on shadertoy.com:
    Common    common.glsl
    Buffer A  buffer_a.glsl
    Image     image.glsl

  Channel wiring:
    Image
      iChannel0  Misc > Buffer A
                filter=linear  wrap=clamp
      iChannel1  Textures > RGBA Noise Medium
                filter=linear  wrap=repeat  vflip=off
                note: same role and size, but different pixel values

  Cannot be reproduced as-is:
    - Buffer A uses scale=0.5; buffers on shadertoy.com are always full resolution
    - image.glsl uses #include; shadertoy.com has no include directive
    - glsl_version is 430; shadertoy.com is GLSL ES 3.0, roughly equivalent to 330

  Worth knowing:
    - Buffer A is read by 2 channels; its filter and wrap are a property of the
      buffer on the site, so set them once
```

Notes are project-specific and actionable only; there is no generic advice
section.

Blockers detected: local texture files (the site has no custom texture upload),
local-only builtins, reduced-resolution buffers, `#include`, and GLSL newer than
ES 3.0. The whole report is in `--json` under `porting`, with a `portable` boolean.
Skip it with `--no-porting`.

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
| `07-path-traced-box` | All five passes at once: a path tracer plus a denoiser |
| `08-cumulonimbus` | Volumetric raymarching: a storm cloud's boiling crown; steer the sun with the mouse, `S` for moonlight |

```bash
shadertoy render -C examples/03-feedback-trail --frame 120
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # everything
pytest -m "not gpu"       # logic only, no GPU needed
```

The suite is split by marker: most tests are pure logic, the rest are marked
`gpu` (skipped when no device is available) or `cpu` (skipped when no software
rasterizer is available). The `cpu` set is a deliberate subset — enough to cover
the GPU-less path and, more usefully, to check the diagnostic parser against a
*real* Mesa compiler rather than a hand-written sample.

Because driver behaviour genuinely differs, the whole suite can be pointed at a
software rasterizer to catch it:

```bash
SHADERTOY_DEVICE=1 SHADERTOY_ALLOW_SOFTWARE=1 pytest
```

That costs about 0.5 s extra. It is worth running before a release: it is how a
crash on every Mesa driver was found, where Mesa reports an array uniform's
length as the number of elements the shader indexes while NVIDIA reports the
full declared length.
