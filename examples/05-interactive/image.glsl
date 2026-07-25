// Simulated input. Nothing here needs a window or a real mouse: input is a
// timeline of operations, so a drag or a press at a specific frame is
// reproducible.
//
//   shadertoy render -C examples/05-interactive --input input.json --frame 80
//   shadertoy render -C examples/05-interactive --count 20 --every 5 --input input.json
//
// Inline works too:
//
//   shadertoy render -C examples/05-interactive \
//     --input '[{"frame":0,"op":"mouse_down","pos":[320,180]}]'
//
// See `shadertoy render --help-input` for the full operation list.
//
// iChannel0 is the keyboard: a 256x3 texture indexed by JavaScript key code.
//   row 0 = held, row 1 = pressed this frame, row 2 = toggle.
#define KEY_SPACE 32
#define KEY_LEFT  37
#define KEY_RIGHT 39
#define KEY_W     87
#define KEY_G     71

float keyHeld(int code)    { return texelFetch(iChannel0, ivec2(code, 0), 0).x; }
float keyPressed(int code) { return texelFetch(iChannel0, ivec2(code, 1), 0).x; }
float keyToggled(int code) { return texelFetch(iChannel0, ivec2(code, 2), 0).x; }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    // Base gradient, hue-shifted by the arrow keys.
    float shift = keyHeld(KEY_RIGHT) - keyHeld(KEY_LEFT);
    vec3 col = 0.5 + 0.5 * cos(6.2831 * (uv.x + shift * 0.25) + vec3(0.0, 2.0, 4.0));
    col *= 0.35 + 0.65 * uv.y;

    // Toggling G switches to a grid overlay.
    if (keyToggled(KEY_G) > 0.5) {
        vec2 g = fract(uv * 16.0);
        float line = min(min(g.x, 1.0 - g.x), min(g.y, 1.0 - g.y));
        col = mix(col, vec3(1.0), smoothstep(0.04, 0.0, line));
    }

    // Holding W inverts.
    col = mix(col, 1.0 - col, keyHeld(KEY_W));

    // Space flashes white only on the frame it is pressed.
    col = mix(col, vec3(1.0), keyPressed(KEY_SPACE) * 0.8);

    // The mouse draws a cursor ring while the button is held.
    // iMouse.z > 0 means "button currently down".
    if (iMouse.z > 0.0) {
        float d = length(fragCoord - iMouse.xy);
        float ring = smoothstep(2.0, 0.0, abs(d - 40.0));
        float dot_ = smoothstep(6.0, 0.0, d);
        col = mix(col, vec3(1.0, 0.95, 0.3), max(ring, dot_));
    }

    fragColor = vec4(col, 1.0);
}
