// Fair-weather cumulus: shared geometry, atmosphere and camera, in kilometres.
// Pass uniforms are supplied as arguments so Common also validates on Shadertoy.
const float PI = 3.14159265359;
const float SUN_AZIMUTH = -1.45;
const float SUN_ELEVATION = 1.0;
const vec3 SUN_RADIANCE = vec3(1.0) * 22.0;
const float MOON_FRACTION = 1.0 / 22.0;
float g_sunAzimuth = SUN_AZIMUTH;
float g_sunElevation = SUN_ELEVATION;
vec3 g_sunRadiance = SUN_RADIANCE;

void setupLighting(vec4 mouse, vec3 res, float moonToggle) {
    if (max(mouse.x, mouse.y) > 0.5) {
        g_sunAzimuth = mix(-PI, PI, clamp(mouse.x / res.x, 0.0, 1.0));
        g_sunElevation = mix(0.02, 1.35, clamp(mouse.y / res.y, 0.0, 1.0));
    }
    g_sunRadiance = SUN_RADIANCE * mix(1.0, MOON_FRACTION, moonToggle);
}

vec3 sunDirection() {
    return vec3(sin(g_sunAzimuth) * cos(g_sunElevation), sin(g_sunElevation),
                cos(g_sunAzimuth) * cos(g_sunElevation));
}

const float BOIL_SPEED = 0.010;
const float INFLATE = 0.48;
const float EXTINCTION = 28.0;
const vec3 ALBEDO = vec3(0.98, 0.985, 0.995);
const vec3 BOX_MIN = vec3(-2.6, 0.8, -1.6);
const vec3 BOX_MAX = vec3(2.6, 3.85, 1.6);

// Rotation-only camera motion permits depth-independent temporal reprojection.
const vec3 CAM_POS = vec3(0.0, 0.05, -5.5);
const float FOCAL = 3.35;
vec2 cameraAngles(float time) {
    return vec2(0.008 * sin(time * 0.051),
                0.375 + 0.004 * sin(time * 0.037 + 1.1));
}

void cameraRay(vec2 fragCoord, vec3 res, vec2 ang, out vec3 ro, out vec3 rd) {
    vec2 uv = (2.0 * fragCoord - res.xy) / res.y;
    vec3 v = normalize(vec3(uv, FOCAL));
    float cp = cos(ang.y), sp = sin(ang.y);
    v = vec3(v.x, cp * v.y + sp * v.z, cp * v.z - sp * v.y);
    float cy = cos(ang.x), sy = sin(ang.x);
    ro = CAM_POS;
    rd = vec3(cy * v.x + sy * v.z, v.y, cy * v.z - sy * v.x);
}

vec2 cameraProject(vec3 rd, vec3 res, vec2 ang) {
    float cy = cos(ang.x), sy = sin(ang.x);
    vec3 v = vec3(cy * rd.x - sy * rd.z, rd.y, cy * rd.z + sy * rd.x);
    float cp = cos(ang.y), sp = sin(ang.y);
    v = vec3(v.x, cp * v.y - sp * v.z, cp * v.z + sp * v.y);
    if (v.z <= 1e-4) return vec2(-1e9);
    return (v.xy * (FOCAL / v.z) * res.y + res.xy) * 0.5;
}

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Conservative ellipsoid distance, including a well-defined negative centre.
float sdEllipsoid(vec3 p, vec3 r) {
    return (length(p / r) - 1.0) * min(r.x, min(r.y, r.z));
}

bool boxRange(vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 d = rd + vec3(lessThan(abs(rd), vec3(1e-8))) * 1e-8;
    vec3 a = (BOX_MIN - ro) / d;
    vec3 b = (BOX_MAX - ro) / d;
    vec3 lo = min(a, b), hi = max(a, b);
    t0 = max(max(lo.x, lo.y), lo.z);
    t1 = min(min(hi.x, hi.y), hi.z);
    return t1 > max(t0, 0.0);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float radicalInverse(int i, int base) {
    float f = 1.0, r = 0.0;
    for (int k = 0; k < 16; k++) {
        if (i <= 0) break;
        f /= float(base);
        r += f * float(i - base * (i / base));
        i /= base;
    }
    return r;
}

vec2 haltonJitter(int frame) {
    int i = frame - 16 * (frame / 16) + 1;
    return vec2(radicalInverse(i, 2), radicalInverse(i, 3)) - 0.5;
}

// Two z slices, not the stock tile's correlated red/green trick: this also works
// with the runtime's independent-channel noise. Linear filtering interpolates xy.
float vnoise(sampler2D tex, vec3 p) {
    vec3 cell = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    vec2 uv = cell.xy + vec2(37.0, 17.0) * cell.z + f.xy + 0.5;
    float a = textureLod(tex, uv / 256.0, 0.0).r;
    float b = textureLod(tex, (uv + vec2(37.0, 17.0)) / 256.0, 0.0).r;
    return mix(a, b, f.z);
}

vec3 vnoise3(sampler2D tex, vec3 p) {
    vec3 cell = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    vec2 uv = cell.xy + vec2(37.0, 17.0) * cell.z + f.xy + 0.5;
    return mix(textureLod(tex, uv / 256.0, 0.0).rgb,
               textureLod(tex, (uv + vec2(37.0, 17.0)) / 256.0, 0.0).rgb, f.z);
}

const mat3 NOISE_ROT = mat3(0.00, 0.80, 0.60,
                           -0.80, 0.36, -0.48,
                           -0.60, -0.48, 0.64);

// The envelope supplies only the broad silhouette and a conservative empty-space
// bound. All fine structure is a continuous density field, not cellular spheres.
float cloudEnvelope(vec3 p, float time) {
    float rise = 0.045 * sin(time * 0.09);
    float d = sdEllipsoid(p - vec3(-0.15, 2.65 + rise, 0.1), vec3(0.85, 0.65, 0.70));
    d = smin(d, sdEllipsoid(p - vec3(0.72, 2.55, 0.05), vec3(0.65, 0.61, 0.63)), 0.25);
    d = smin(d, sdEllipsoid(p - vec3(-0.80, 2.18, -0.05), vec3(0.73, 0.57, 0.65)), 0.3);
    d = smin(d, sdEllipsoid(p - vec3(0.08, 2.04, -0.18), vec3(1.22, 0.63, 0.82)), 0.35);
    d = smin(d, sdEllipsoid(p - vec3(1.05, 1.92, 0.02), vec3(0.58, 0.35, 0.52)), 0.25);
    return d;
}

float crownCondensation(float height) {
    // Quintic blend joins density and its first two derivatives smoothly.
    float h = clamp((height - 2.15) / 0.95, 0.0, 1.0);
    return clamp(h * h * h * (h * (h * 6.0 - 15.0) + 10.0), 0.0, 1.0);
}

// A dense crown above dilute, wind-torn side/base layers. View and shadow rays
// share the same density; opacity emerges from the depth of material traversed.
float cloudDensity(sampler2D tex, vec3 p, float time, float sdf) {
    if (sdf >= INFLATE) return 0.0;
    float crown = smoothstep(1.7, 2.8, p.y);
    float evaporating = 1.0 - crownCondensation(p.y);
    if (sdf < -0.85 && evaporating == 0.0) return 1.0;
    float erosion = mix(0.85, 0.35, crown);
    float condensed = mix(-0.075, 0.075, crown);
    float baseDepth = -sdf + condensed;
    if (baseDepth + 0.50 * erosion <= -0.12) return 0.0;
    if (baseDepth - 0.50 * erosion >= 0.55) return 1.0;
    if (evaporating == 0.0 && baseDepth + 0.50 * erosion <= 0.0) return 0.0;
    if (evaporating == 0.0 && baseDepth - 0.50 * erosion >= 0.18) return 1.0;
    vec3 q = p - vec3(time * 0.004, time * BOIL_SPEED, 0.0);
    q += (vnoise3(tex, q * 1.5 + 17.0) - 0.5) * 0.25;
    q *= 3.8;
    float f = 0.52 * vnoise(tex, q);
    // The remaining octaves lie in [0, 0.48]. These bounds skip only samples
    // guaranteed to be empty or fully dense, without discarding visible detail.
    if (baseDepth + (f + 0.48 - 0.50) * erosion <= -0.12) return 0.0;
    if (baseDepth + (f - 0.50) * erosion >= 0.55) return 1.0;
    if (evaporating == 0.0 && baseDepth + (f + 0.48 - 0.50) * erosion <= 0.0) return 0.0;
    if (evaporating == 0.0 && baseDepth + (f - 0.50) * erosion >= 0.18) return 1.0;
    q = NOISE_ROT * q * 2.03 + 7.1;
    f += 0.26 * vnoise(tex, q);
    q = NOISE_ROT * q * 2.03 + 7.1;
    f += 0.13 * vnoise(tex, q);
    q = NOISE_ROT * q * 2.03 + 7.1;
    f += 0.065 * vnoise(tex, q);
    q = NOISE_ROT * q * 2.03 + 7.1;
    f += 0.025 * vnoise(tex, q);
    float depth = baseDepth + (f - 0.50) * erosion;
    if (depth <= -0.12) return 0.0;
    if (depth >= 0.55) return 1.0;
    if (evaporating == 0.0 && depth <= 0.0) return 0.0;
    if (evaporating == 0.0 && depth >= 0.18) return 1.0;
    if (depth < 0.16 && evaporating < 1.0) {
        vec3 micro = p * vec3(58.0, 73.0, 64.0) - time * vec3(0.24, 0.65, 0.0);
        float ragged = 0.65 * vnoise(tex, micro) + 0.35 * vnoise(tex, NOISE_ROT * micro * 2.1);
        depth -= 0.055 * (1.0 - ragged) * (1.0 - smoothstep(0.03, 0.16, depth));
    }
    float density = smoothstep(0.0, mix(0.05, 0.025, crown), depth);
    if (depth < 0.18 && evaporating < 1.0) {
        vec3 w = p - vec3(time * 0.008, time * BOIL_SPEED, 0.0);
        w.x += 0.35 * w.y;
        w *= vec3(9.0, 23.0, 18.0);
        float wisps = 0.65 * vnoise(tex, w) + 0.35 * vnoise(tex, NOISE_ROT * w * 2.1);
        float fringe = smoothstep(0.32, 0.72, wisps);
        float fringeDepth = mix(0.18, 0.10, crown);
        density *= mix(fringe * fringe, 1.0, smoothstep(0.0, fringeDepth, depth));
    }
    if (evaporating > 0.0) {
        vec3 flow = p - time * vec3(0.008, BOIL_SPEED, 0.0);
        flow += 0.16 * (vnoise3(tex, flow * 2.0 + 31.7) - 0.5);
        flow.y -= 0.15 * flow.x;
        float entrainment = vnoise(tex, flow * vec3(3.0, 6.0, 4.0) + 43.1);
        float mixingDepth = baseDepth + (f - 0.5) * erosion + 0.12 - 0.12 * entrainment;
        if (evaporating == 1.0 && mixingDepth <= 0.0) return 0.0;
        flow *= vec3(8.0, 18.0, 12.0);
        float strands = 0.0;
        for (int i = 0; i < 3; i++) {
            vec2 sheets = 1.0 - smoothstep(vec2(0.025), vec2(0.20),
                abs(vnoise3(tex, flow).xy - 0.5));
            float reach = i == 0 ? 0.85 : (i == 1 ? 0.65 : 0.40);
            strands = max(strands, reach * sheets.x * sheets.y);
            flow = NOISE_ROT * flow * 2.03 + 17.1;
        }
        float occupied = smoothstep(0.0, 0.03, mixingDepth - 0.15 * (1.0 - strands));
        float liquid = smoothstep(0.0, 0.30, mixingDepth);
        // Only the mixing layer is dilute; a dense interior prevents backlight
        // from turning the entire lower half into a bright translucent slab.
        float core = smoothstep(0.18, 0.55, baseDepth + (f - 0.5) * erosion);
        density = mix(density, mix(0.18, 1.0, core) * occupied * liquid * sqrt(liquid), evaporating);
    }
    return density;
}

// Analytic single-scattering atmosphere with a small diffuse sky term.
const vec3 BETA_R = vec3(5.802e-3, 13.558e-3, 33.100e-3);
const float BETA_M = 3.996e-3;
const vec3 BETA_O = vec3(0.650e-3, 1.881e-3, 0.085e-3);
const vec3 ZENITH_OD = BETA_R * 8.0 + vec3(BETA_M * 1.2) + BETA_O * 15.0;
float airMass(float cosZenith) {
    return 38.0 / (37.0 * clamp(cosZenith, 0.0, 1.0) + 1.0);
}
vec3 sunTransmittance(vec3 sd) {
    return exp(-ZENITH_OD * airMass(sd.y));
}
float hg(float g, float mu) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}
vec3 skyRadiance(vec3 rd, vec3 sd) {
    float mu = clamp(dot(rd, sd), -1.0, 1.0);
    float phaseR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    float phaseM = hg(0.76, mu);
    vec3 col = vec3(0.0), viewT = vec3(1.0);
    float h0 = 0.0;
    // Integrate atmospheric layers instead of tinting the entire sky with the
    // ground-level sunset beam. High-altitude Rayleigh scatter stays blue.
    for (int i = 0; i < 8; i++) {
        float h1 = exp2(float(i)) * 0.65;
        float h = (h0 + h1) * 0.5;
        float r = 8.0 * (exp(-h0 / 8.0) - exp(-h1 / 8.0));
        float m = 1.2 * (exp(-h0 / 1.2) - exp(-h1 / 1.2));
        float o = 15.0 * (smoothstep(10.0, 40.0, h1) - smoothstep(10.0, 40.0, h0));
        vec3 od = (BETA_R * r + vec3(BETA_M * m) + BETA_O * o) * airMass(rd.y);
        vec3 sunOD = BETA_R * 8.0 * exp(-h / 8.0) + vec3(BETA_M * 1.2 * exp(-h / 1.2)) +
                     BETA_O * 15.0 * (1.0 - smoothstep(10.0, 40.0, h));
        vec3 source = (BETA_R * r * phaseR + vec3(BETA_M * m * phaseM)) * airMass(rd.y);
        vec3 stepT = exp(-od);
        col += viewT * (1.0 - stepT) * source / max(od, vec3(1e-6)) * exp(-sunOD * airMass(sd.y));
        viewT *= stepT;
        h0 = h1;
    }
    vec3 diffuse = 0.03 * (BETA_R / BETA_R.b) * (1.0 - exp(-ZENITH_OD * airMass(rd.y)));
    return g_sunRadiance * (col + diffuse);
}
vec3 sunDisc(vec3 rd, vec3 sd) {
    float mu = dot(rd, sd);
    vec3 t = g_sunRadiance * sunTransmittance(sd);
    return t * (240.0 * smoothstep(0.99988, 0.99996, mu) +
                0.85 * pow(max(mu, 0.0), 2200.0) +
                0.020 * pow(max(mu, 0.0), 130.0));
}
float cloudPhase(float mu, float ecc) {
    return mix(hg(0.80 * ecc, mu), hg(-0.35 * ecc, mu), 0.30);
}

// Approximate multiple scattering with decreasing extinction and anisotropy.
// All orders reuse one shadow march; no nested path tracing or offline baking.
vec3 sunScatter(float opticalDepth, float mu, vec3 sunCol) {
    vec3 l = vec3(0.0);
    float a = 1.0, b = 1.0, c = 1.0;
    for (int i = 0; i < 5; i++) {
        l += b * exp(-a * opticalDepth) * cloudPhase(mu, c);
        a *= 0.40;
        b *= 0.65;
        c *= 0.60;
    }
    // Residual multiple scattering is diffuse, not a second forward peak.
    l += 0.30 * exp(-0.18 * opticalDepth) * cloudPhase(mu, 0.40);
    return l * sunCol;
}
vec3 ambientScatter(vec3 skyLight, float skyOD, float sunOD, float refill) {
    float lum = dot(skyLight, vec3(0.2126, 0.7152, 0.0722));
    vec3 amb = mix(skyLight, vec3(lum), clamp(0.55 + skyOD * 0.30 + sunOD * 0.10, 0.55, 0.90));
    float f = 0.22 * refill;
    return amb * (f + (1.0 - f) / (1.0 + skyOD * 0.36));
}

const float EXPOSURE = 0.39;
const float KEY_AZIMUTH = -1.45;
const float KEY_ELEVATION = 1.0;
const vec3 KEY_RADIANCE = vec3(1.0) * 22.0;
const float SKY_GAIN = 0.45;
vec3 keySunDirection() {
    return vec3(sin(KEY_AZIMUTH) * cos(KEY_ELEVATION), sin(KEY_ELEVATION),
                cos(KEY_AZIMUTH) * cos(KEY_ELEVATION));
}
float keyLuminance(vec3 radiance, vec3 sd, vec3 rdC) {
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
    vec3 sunCol = radiance * sunTransmittance(sd);
    float ph = cloudPhase(clamp(dot(rdC, sd), -1.0, 1.0), 1.0);
    ph = ph / (1.0 + 2.2 * max(ph - 0.15, 0.0)) + 0.08;
    vec3 sky = skyRadiance(vec3(0.0, 1.0, 0.0), sd) *
               (dot(radiance, LUMA) / dot(g_sunRadiance, LUMA));
    return dot(sunCol, LUMA) * ph + dot(sky, LUMA) * 1.2;
}
float sceneExposure(vec3 sd, vec3 rdC) {
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
    float rGeo = keyLuminance(KEY_RADIANCE, keySunDirection(), rdC) /
                 max(keyLuminance(KEY_RADIANCE, sd, rdC), 1e-5);
    float rRad = dot(KEY_RADIANCE, LUMA) / max(dot(g_sunRadiance, LUMA), 1e-5);
    return EXPOSURE * pow(rGeo, rGeo > 1.0 ? 0.85 : 1.0) *
                      pow(rRad, rRad > 1.0 ? 0.50 : 1.0);
}
float nightness() {
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
    float stops = log2(dot(KEY_RADIANCE, LUMA) / max(dot(g_sunRadiance, LUMA), 1e-5));
    return smoothstep(1.5, 4.0, stops);
}
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 tonemap(vec3 x, float skyness) {
    float l = max(dot(x, vec3(0.2126, 0.7152, 0.0722)), 1e-5);
    float ln = aces(vec3(l)).x;
    vec3 c = mix(x * (ln / l), aces(x), smoothstep(0.85, 1.0, ln));
    c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, 1.0 + 0.35 * skyness);
    c *= vec3(0.99, 1.0, 1.025);
    return clamp(c, 0.0, 1.0);
}
