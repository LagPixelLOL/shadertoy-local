// Shared code. This file is prepended to every pass, exactly like
// Shadertoy's "Common" tab.

const float PI = 3.14159265359;

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// Signed distance to a sphere.
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

// Signed distance to a rounded box.
float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Smooth minimum, for blending shapes together.
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
