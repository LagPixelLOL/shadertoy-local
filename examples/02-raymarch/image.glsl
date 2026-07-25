// Raymarched signed-distance scene. Demonstrates using common.glsl for
// shared helpers (sdSphere, sdRoundBox, smin, rot all come from there).

// Scene distance function: a sphere blended into a rotating rounded box.
float map(vec3 p) {
    vec3 q = p;
    q.xz = rot(iTime * 0.5) * q.xz;
    float box = sdRoundBox(q, vec3(0.6), 0.1);
    float ball = sdSphere(p - vec3(0.0, 0.7 + 0.2 * sin(iTime * 2.0), 0.0), 0.35);
    float ground = p.y + 1.0;
    return smin(smin(box, ball, 0.25), ground, 0.1);
}

// Gradient of the distance field, i.e. the surface normal.
vec3 calcNormal(vec3 p) {
    const vec2 e = vec2(0.0005, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;

    vec3 ro = vec3(0.0, 0.4, 3.0);          // ray origin (camera)
    vec3 rd = normalize(vec3(uv, -1.6));    // ray direction

    // Sphere-trace the distance field.
    float t = 0.0;
    bool hit = false;
    for (int i = 0; i < 96; ++i) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001) { hit = true; break; }
        if (t > 20.0) break;
        t += d;
    }

    vec3 col = vec3(0.05, 0.07, 0.10);      // background
    if (hit) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 lightDir = normalize(vec3(0.7, 0.8, 0.4));

        float diffuse = max(dot(n, lightDir), 0.0);
        float ambient = 0.15 + 0.35 * (0.5 + 0.5 * n.y);
        float spec = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), 32.0);

        col = vec3(0.55, 0.65, 0.85) * (ambient + diffuse) + spec * 0.6;
        col *= exp(-0.04 * t * t);          // distance fog
    }

    col = pow(col, vec3(0.4545));           // linear -> sRGB
    fragColor = vec4(col, 1.0);
}
