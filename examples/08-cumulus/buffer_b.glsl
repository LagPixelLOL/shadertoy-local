// Buffer B -- variance-clipped exponential temporal resolve.
// iChannel0: Buffer A (radiance RGB, transmittance A), nearest.
// iChannel1: previous Buffer B, linear/clamp; iChannel2: Keyboard.
// Output has the same cloud layout except texel (0, 0), which stores lighting
// state. Image excludes it from composition and shaft sampling.
// Rotation-only camera motion makes reprojection independent of volume depth;
// variance clipping handles cloud evolution, which is not reprojected.

// A 16-frame time constant, not an exact average of the last sixteen frames.
#define BLEND 0.0625

// Width in standard deviations of the current 3x3 neighbourhood.
#define CLIP_SIGMA 1.25

vec4 fetchA(ivec2 q) {
    return texelFetch(iChannel0, clamp(q, ivec2(0), ivec2(iResolution.xy) - 1), 0);
}

// Nine bilinear taps implement Catmull-Rom, avoiding the cumulative blur of
// repeatedly resampling moving history with a single bilinear tap. Its negative
// lobes require both variance clipping and physical output bounds below.
vec4 historyCatmullRom(vec2 pos, vec2 res) {
    vec2 centre = floor(pos - 0.5) + 0.5;
    vec2 f = pos - centre;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 p0 = (centre - 1.0) / res;
    vec2 p3 = (centre + 2.0) / res;
    vec2 p12 = (centre + w2 / w12) / res;
    return
        texture(iChannel1, vec2(p0.x,  p0.y))  * (w0.x  * w0.y) +
        texture(iChannel1, vec2(p12.x, p0.y))  * (w12.x * w0.y) +
        texture(iChannel1, vec2(p3.x,  p0.y))  * (w3.x  * w0.y) +
        texture(iChannel1, vec2(p0.x,  p12.y)) * (w0.x  * w12.y) +
        texture(iChannel1, vec2(p12.x, p12.y)) * (w12.x * w12.y) +
        texture(iChannel1, vec2(p3.x,  p12.y)) * (w3.x  * w12.y) +
        texture(iChannel1, vec2(p0.x,  p3.y))  * (w0.x  * w3.y) +
        texture(iChannel1, vec2(p12.x, p3.y))  * (w12.x * w3.y) +
        texture(iChannel1, vec2(p3.x,  p3.y))  * (w3.x  * w3.y);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 q = ivec2(fragCoord);
    float moon = texelFetch(iChannel2, ivec2(83, 2), 0).x;
    setupLighting(iMouse, iResolution, moon);
    vec4 lighting = vec4(sunDirection() * 0.5 + 0.5, moon);
    if (all(equal(q, ivec2(0)))) {
        fragColor = lighting;
        return;
    }
    vec4 current = fetchA(q);

    // First and second moments of the 3x3 neighbourhood. Moments rather than a
    // min/max box: a single bright sample from a fine wisp would widen a box
    // enough to let a whole frame of ghost through, while it barely moves the
    // standard deviation.
    vec4 m1 = vec4(0.0), m2 = vec4(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec4 s = fetchA(q + ivec2(x, y));
            m1 += s;
            m2 += s * s;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    vec4 sigma = sqrt(max(m2 - m1 * m1, vec4(0.0)));
    // A persistent fine edge must not be clipped away merely for being narrower
    // than its neighbourhood. Matching current/history samples remain intact.
    vec4 lo = min(m1 - CLIP_SIGMA * sigma, current);
    vec4 hi = max(m1 + CLIP_SIGMA * sigma, current);

    vec4 resolved = current;

    vec4 prevLighting = texelFetch(iChannel1, ivec2(0), 0);
    if (iFrame > 0 && all(lessThan(abs(lighting - prevLighting), vec4(1e-5)))) {
        vec2 ang = cameraAngles(iTime);
        vec2 prevAng = cameraAngles(iTime - iTimeDelta);
        vec3 ro, rd;
        cameraRay(fragCoord, iResolution, ang, ro, rd);
        vec2 prevCoord = cameraProject(rd, iResolution, prevAng);

        // Two and a half pixels in from the edge, so no Catmull-Rom tap can
        // reach outside the frame and clamp a wrong sample into the history.
        if (all(greaterThan(prevCoord, vec2(2.5))) &&
            all(lessThan(prevCoord, iResolution.xy - 2.5))) {
            vec4 history = historyCatmullRom(prevCoord, iResolution.xy);
            resolved = mix(clamp(history, lo, hi), current, BLEND);
        }
    }

    fragColor = vec4(max(resolved.rgb, vec3(0.0)), clamp(resolved.a, 0.0, 1.0));
}
