// Buffer A: a feedback accumulator.
//
// iChannel0 is wired to buffer_a -- itself -- so sampling it reads this
// buffer's contents from the PREVIOUS frame. That self-reference is what makes
// trails, fluid sims and cellular automata possible.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    // Last frame's state.
    vec3 prev = texture(iChannel0, uv).rgb;

    // Two orbiting emitters, in aspect-corrected space.
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float t = iTime;
    vec2 a = 0.55 * vec2(cos(t * 1.1), sin(t * 1.7));
    vec2 b = 0.55 * vec2(cos(t * 1.7 + 2.0), sin(t * 1.1 + 2.0));

    float ea = smoothstep(0.035, 0.0, length(p - a));
    float eb = smoothstep(0.035, 0.0, length(p - b));

    vec3 emit = ea * vec3(1.0, 0.35, 0.15) + eb * vec3(0.15, 0.5, 1.0);

    // Decay the old state, then add the new emission. The decay factor sets
    // the trail length; tune it and re-render to see the difference.
    vec3 col = prev * 0.965 + emit;

    fragColor = vec4(col, 1.0);
}
