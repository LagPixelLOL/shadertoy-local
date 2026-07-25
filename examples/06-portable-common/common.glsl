// Portable Common tab.
//
// Nothing outside a #define names a pass-specific uniform, so this file
// validates cleanly in shadertoy.com's Common tab -- which checks Common
// standalone against a minimal header where only iDate and iSampleRate exist.
//
// Keeping Common clean is not cosmetic: on the site, ~30 lines of spurious
// "undeclared identifier" noise will hide a genuine typo in this file.

// Every pass-specific value a helper might want, in one struct.
struct ST {
    vec3  res;      // iResolution
    float time;     // iTime
    float dt;       // iTimeDelta
    int   frame;    // iFrame
    vec4  mouse;    // iMouse
};

// A macro body is only compiled where it is EXPANDED. Since this is expanded
// inside a pass -- where the uniforms genuinely exist -- naming them here is
// legal, and the site never parses it while validating Common on its own.
// This is the one place Common may mention them.
#define ST_CAPTURE ST(iResolution, iTime, iTimeDelta, iFrame, iMouse)

// ---- helpers below take state as a parameter, never from a uniform ----

float aspect(ST u) {
    return u.res.x / u.res.y;
}

// Aspect-corrected coordinates centred on the screen.
vec2 centred(ST u, vec2 fragCoord) {
    return (2.0 * fragCoord - u.res.xy) / u.res.y;
}

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(6.2831 * (t + vec3(0.0, 0.33, 0.67)));
}

// Normalised cursor position, or the screen centre when the button is up.
vec2 cursor(ST u) {
    return u.mouse.z > 0.0 ? u.mouse.xy / u.res.xy : vec2(0.5);
}

float sdCircle(vec2 p, float r) {
    return length(p) - r;
}
