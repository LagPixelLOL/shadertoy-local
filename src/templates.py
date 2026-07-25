"""Project scaffolding for ``shadertoy init``.

Templates are intentionally small and correct rather than impressive: their job
is to give a starting point that compiles, renders something non-uniform, and
demonstrates the config wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

_BASIC_IMAGE = """// Image pass. Shadertoy semantics: fragCoord has its origin at the
// bottom-left, and iResolution.xy is the output size in pixels.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    // Animated colour ramp.
    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));

    fragColor = vec4(col, 1.0);
}
"""

_COMMON = """// Shared code, prepended to every pass. Put helpers here.

float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}
"""

_FEEDBACK_BUFFER = """// Buffer A: a feedback buffer. iChannel0 is bound to Buffer A itself,
// so sampling it reads *last* frame's contents.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    vec4 prev = texture(iChannel0, uv);

    // Paint a moving dot into the buffer.
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec2 centre = 0.3 * vec2(cos(iTime), sin(iTime * 1.3));
    float dot_ = smoothstep(0.02, 0.0, length(p - centre));

    // Decay what was already there, then add the new dot: a motion trail.
    fragColor = vec4(max(prev.rgb * 0.96, vec3(dot_)), 1.0);
}
"""

_FEEDBACK_IMAGE = """// Image pass: displays Buffer A. Reading another buffer gives you that
// buffer's output from the *current* frame.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = texture(iChannel0, uv).rgb;
    col = pow(col, vec3(0.75));               // lift the trail
    fragColor = vec4(col * vec3(0.4, 0.8, 1.0), 1.0);
}
"""

_INPUT_IMAGE = """// Demonstrates simulated input. Drive it with:
//   shadertoy render --mouse 320,180 --key space
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = vec3(uv, 0.3);

    // iMouse.xy is the cursor; iMouse.z > 0 while the button is held.
    if (iMouse.z > 0.0) {
        float d = length(fragCoord - iMouse.xy);
        col = mix(vec3(1.0, 0.9, 0.2), col, smoothstep(20.0, 22.0, d));
    }

    // Keyboard: row 0 of the keyboard texture is "key is held".
    // 32 is the JavaScript key code for space.
    float space = texelFetch(iChannel0, ivec2(32, 0), 0).x;
    col = mix(col, vec3(1.0) - col, space);

    fragColor = vec4(col, 1.0);
}
"""


def _config(data: dict) -> str:
    return json.dumps(data, indent=2) + "\n"


TEMPLATES: dict[str, dict[str, str]] = {
    "basic": {
        "image.glsl": _BASIC_IMAGE,
    },
    "common": {
        "image.glsl": _BASIC_IMAGE,
        "common.glsl": _COMMON,
    },
    "feedback": {
        "buffer_a.glsl": _FEEDBACK_BUFFER,
        "image.glsl": _FEEDBACK_IMAGE,
        "shadertoy.json": _config(
            {
                "defaults": {"width": 640, "height": 360, "fps": 60},
                "buffer_a": {
                    "channels": {"0": {"type": "buffer", "source": "buffer_a"}}
                },
                "image": {
                    "channels": {"0": {"type": "buffer", "source": "buffer_a"}}
                },
            }
        ),
    },
    "input": {
        "image.glsl": _INPUT_IMAGE,
        "shadertoy.json": _config(
            {
                "defaults": {"width": 640, "height": 360},
                "image": {"channels": {"0": {"type": "keyboard"}}},
            }
        ),
    },
}

DEFAULT_TEMPLATE = "basic"


def scaffold(root: Path, template: str = DEFAULT_TEMPLATE, force: bool = False) -> list[Path]:
    """Write a template into *root*, returning the files created."""
    if template not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES))
        raise ValueError(f"unknown template {template!r}. Available: {available}")

    root = Path(root)
    files = TEMPLATES[template]

    if not force:
        clashes = [name for name in files if (root / name).exists()]
        if clashes:
            raise FileExistsError(
                f"refusing to overwrite existing file(s): {', '.join(sorted(clashes))}. "
                f"Pass --force to overwrite."
            )

    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in sorted(files.items()):
        path = root / name
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
