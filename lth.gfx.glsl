#version 430
/*
    "LTH" by gam0022 & kazpulse
    PC 8K Intro at Akiba Executable Party 2026

    - MRT0 前フレームの MRT3 へ ポストプロセス（FXAA と Bloom）を適用した最終表示色
    - MRT1 LTC用のLUTとランダムノイズ（初期化時に生成して更新しない）
    - MRT2 発光模様 8x2の合計16枚のアトラス（模様領域はライトと同じ縦横比）
    - MRT3 RGB: 3Dシーンの描画結果, A: Bloom マスク
*/
layout(binding = 0) uniform sampler2D backBuffer0;
layout(binding = 1) uniform sampler2D backBuffer1;
layout(binding = 2) uniform sampler2D backBuffer2;
layout(binding = 3) uniform sampler2D backBuffer3;
layout(location = 0) uniform int waveOutPosition;
#define NUM_SOUND_BUFFER_SAMPLES 0x1000000
const float SAMPLES_PER_SECOND = 44100.0;
#if defined(EXPORT_EXECUTABLE)
#pragma work_around_begin:layout(std430,binding=0)buffer ssbo{vec2 %s[];};
vec2 waveOutSamples[];
#pragma work_around_end
#else
layout(std430, binding = 0) buffer _{ vec2 waveOutSamples[]; };
#endif
#if defined(EXPORT_EXECUTABLE)
#define resolution vec2(SCREEN_XRESO, SCREEN_YRESO)
float time = float(waveOutPosition) / SAMPLES_PER_SECOND;
#else
layout(location = 2) uniform float time;
layout(location = 3) uniform vec2 resolution;
#endif

// debug用のuniform
#if !defined(EXPORT_EXECUTABLE)

// カメラ
layout(location = 6) uniform float tanFovY;
layout(location = 7) uniform mat4 cameraInWorld;

// パラメーター
layout(location = 16) uniform int debugLightCount;  // @ui slider (min=1, max=12, default=12)
layout(location = 17) uniform float debugParam1;  // @ui slider (min=0, max=1, default=0)
layout(location = 18) uniform float debugParam2;  // @ui slider (min=0, max=1, default=0)
#endif

layout(location = 0) out vec4 outColor0;
layout(location = 1) out vec4 outColor1;
layout(location = 2) out vec4 outColor2;
layout(location = 3) out vec4 outColor3;

float beat, beatTau, beatPhase;
int nBeat;
// MRTのフレーム遅延を補正した音声サンプル位置
int adjustedWaveOutPosition;

#define saturate(x) clamp(x, 0., 1.)
#define tri(x) (1. - 4. * abs(fract(x) - .5))
#define phase(x) (floor(x) + .5 + .5 * cos(TAU * .5 * exp(-5. * fract(x))))

// 全材質を metallic=1.0 とみなし、不要になる拡散反射と metallic の計算を省く。0 で従来の可変 metallic 経路に戻す
#define METALLIC_ONE 1

// 全ライトを twoSided=false の片面発光とみなし、両面発光用のデータと分岐を省く。0 で従来の可変 twoSided 経路に戻す
#define ONE_SIDED_LIGHTS 1

mat2 rot(float a) { return mat2(cos(a), sin(a), -sin(a), cos(a)); }
mat3 rotationXYZ(vec3 rotation)
{
    vec3 c = cos(rotation);
    vec3 s = sin(rotation);
    mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, c.x, s.x, 0.0, -s.x, c.x);
    mat3 ry = mat3(c.y, 0.0, -s.y, 0.0, 1.0, 0.0, s.y, 0.0, c.y);
    mat3 rz = mat3(c.z, s.z, 0.0, -s.z, c.z, 0.0, 0.0, 0.0, 1.0);
    return rz * ry * rx;
}

float t = 0.0, blend = 0.0;
bool inRange(float x, float start, float end)
{
    t = x - start;
    blend = t / (end - start);
    return x >= start && x < end;
}

const float TAU = 6.28318530718;
const float BLOOM_INTENSITY = 0.9;
const int EMISSION_ATLAS_COLUMNS = 8;
const int EMISSION_ATLAS_ROWS = 2;

// ライトの数とレイアウト設定
const int LIGHT_COUNT = 12;
const vec3 LIGHT_SCALE = vec3(0.5, 1.5, 0.10);
const float LIGHT_HEIGHT = 1.5;

const int LTC_LUT_SIZE = 8;
const float LTC_LUT_NORMALIZATION[64] = float[64](
    1.000, 1.000, 0.999, 0.954, 0.855, 0.678, 0.476, 0.307,
    1.000, 1.000, 0.995, 0.956, 0.852, 0.677, 0.479, 0.312,
    1.000, 1.000, 0.993, 0.950, 0.845, 0.675, 0.488, 0.327,
    1.000, 1.000, 0.992, 0.943, 0.832, 0.674, 0.506, 0.356,
    1.000, 1.000, 0.986, 0.930, 0.814, 0.679, 0.538, 0.403,
    1.000, 0.999, 0.973, 0.900, 0.804, 0.704, 0.595, 0.481,
    1.000, 0.993, 0.926, 0.861, 0.833, 0.783, 0.704, 0.620,
    0.987, 0.939, 0.943, 0.944, 0.944, 0.944, 0.943, 0.943
);

// サイズ削減のため、64個のvec4からなる行列LUTをfloat配列に平坦化し、参照時にvec4へ復元する
const float LTC_LUT_MATRIX[256] = float[256](
    0.000200, 0.000000, 1.000000, 0.000000, 0.040821, 0.000000, 1.000000, 0.000000, 0.163499, 0.000000, 1.000000, 0.000000, 0.359810, 0.000000, 1.000000, 0.000000,
    0.608219, 0.000000, 1.000000, 0.000000, 0.849327, 0.000000, 1.000000, 0.000000, 1.026876, 0.000000, 1.000000, 0.000000, 1.127918, 0.000000, 1.000000, 0.000000,
    0.000200, -0.000046, 1.052217, 0.228243, 0.040821, -0.009316, 1.052066, 0.228242, 0.163218, -0.036566, 1.051517, 0.228141, 0.360871, -0.077435, 1.049110, 0.226851,
    0.608870, -0.112986, 1.040456, 0.216178, 0.850681, -0.110371, 1.022363, 0.171798, 1.033758, -0.063836, 1.011022, 0.089896, 1.130218, 0.000062, 1.000010, -0.000080,
    0.000200, -0.000096, 1.232269, 0.481573, 0.040829, -0.019653, 1.232083, 0.481559, 0.163323, -0.077152, 1.229494, 0.481157, 0.361362, -0.161748, 1.215935, 0.476966,
    0.602530, -0.224893, 1.149731, 0.444834, 0.832866, -0.211716, 1.066321, 0.339254, 1.033638, -0.123274, 1.025543, 0.170803, 1.137644, 0.000154, 0.999953, -0.000082,
    0.000199, -0.000159, 1.627474, 0.797472, 0.040836, -0.032524, 1.635685, 0.797401, 0.164108, -0.127318, 1.628042, 0.795956, 0.365844, -0.258932, 1.575327, 0.783120,
    0.602895, -0.339455, 1.372632, 0.710306, 0.825683, -0.302224, 1.158343, 0.512880, 1.030563, -0.173519, 1.049980, 0.243539, 1.151713, 0.000105, 0.999957, -0.000055,
    0.000198, -0.000248, 2.529443, 1.253956, 0.040880, -0.051100, 2.570697, 1.253684, 0.166706, -0.198049, 2.542881, 1.248409, 0.380361, -0.384441, 2.334701, 1.206127,
    0.617299, -0.459699, 1.772357, 1.025249, 0.835445, -0.380695, 1.304201, 0.680722, 1.040850, -0.213042, 1.089431, 0.306573, 1.176060, 0.000010, 1.000023, -0.000021,
    0.000187, -0.000388, 4.301116, 2.076506, 0.041070, -0.084546, 5.305590, 2.075270, 0.175521, -0.321638, 5.161014, 2.051541, 0.413187, -0.560077, 4.055352, 1.869406,
    0.665130, -0.580464, 2.432246, 1.376237, 0.870094, -0.441204, 1.506300, 0.819691, 1.068578, -0.241436, 1.137736, 0.354430, 1.219174, -0.000534, 1.000063, 0.000162,
    0.000144, -0.000631, 6.468958, 4.381222, 0.042200, -0.177353, 20.085962, 4.369584, 0.209979, -0.606367, 16.868328, 4.106197, 0.500901, -0.795643, 8.099620, 2.893355,
    0.769693, -0.688227, 3.439625, 1.659662, 0.955036, -0.470746, 1.760819, 0.883184, 1.134532, -0.254869, 1.190455, 0.377490, 1.308330, -0.002944, 0.999844, 0.001275,
    0.000102, -0.118482, 1184.816772, 1184.816284, 0.319603, -3.097308, 7133.888672, 23.645655, 0.546638, -1.442077, 133.966263, 8.593670, 0.837342, -1.129903, 20.226521, 3.658263,
    1.004258, -0.809128, 5.700182, 1.768790, 1.117055, -0.531464, 2.269246, 0.923613, 1.336692, -0.312752, 1.241618, 0.414403, 1.670654, -0.056394, 0.998759, 0.033416
);

vec4 referenceLUTTexel(ivec2 pixel)
{
    if (pixel.y < 0 || pixel.y >= LTC_LUT_SIZE || pixel.x < 0 || pixel.x >= LTC_LUT_SIZE * 2)
        return vec4(0.0);
    int index = pixel.y * LTC_LUT_SIZE + pixel.x % LTC_LUT_SIZE;
    return pixel.x < LTC_LUT_SIZE
        ? vec4(LTC_LUT_NORMALIZATION[index])
        : vec4(
            LTC_LUT_MATRIX[index * 4],
            LTC_LUT_MATRIX[index * 4 + 1],
            LTC_LUT_MATRIX[index * 4 + 2],
            LTC_LUT_MATRIX[index * 4 + 3]
        );
}

vec2 referenceLUTUv(vec2 uv, bool matrix)
{
    vec2 cacheSize = vec2(textureSize(backBuffer1, 0));
    vec2 texel = saturate(uv) * float(LTC_LUT_SIZE - 1) + 0.5;
    if (matrix)
        texel.x += float(LTC_LUT_SIZE);
    return texel / cacheSize;
}

const uint C_HASH = 20242024u;

vec4 hash42(vec2 p)
{
    uvec4 x = floatBitsToUint(vec4(p, 1.0, 2.0));
    x = C_HASH * ((x >> 8u) ^ x.yzwx);
    x = C_HASH * ((x >> 8u) ^ x.yzwx);
    x = C_HASH * ((x >> 8u) ^ x.yzwx);
    return vec4(x) / float(0xffffffffu);
}

vec4 randomTexel(ivec2 pixel)
{
    return hash42(vec2(pixel));
}

vec4 backBuffer1Texel(ivec2 pixel)
{
    if (pixel.x < LTC_LUT_SIZE * 2 && pixel.y < LTC_LUT_SIZE)
        return referenceLUTTexel(pixel);
    return randomTexel(pixel);
}

vec4 valueNoise(vec2 position)
{
    vec2 cacheSize = vec2(textureSize(backBuffer1, 0));
    vec2 noiseMin = vec2(1.5, float(LTC_LUT_SIZE) + 1.5);
    vec2 noiseMax = cacheSize - vec2(2.5);
    vec2 noiseExtent = max(noiseMax - noiseMin, vec2(1.0));
    vec2 period = 2.0 * noiseExtent;
    vec2 cell = floor(position);
    vec2 f = fract(position);
    vec2 cubic = f * f * (3.0 - 2.0 * f);
    vec2 smoothPosition = cell + cubic;
    vec2 texel = noiseMin + noiseExtent
               - abs(mod(smoothPosition, period) - noiseExtent);
    return 2.0 * textureLod(backBuffer1, texel / cacheSize, 0.0) - 1.0;
}

vec4 fbm(vec2 position)
{
    mat2 octaveTransform = mat2(1.6, 1.2, -1.2, 1.6);
    vec4 noise = 0.5 * valueNoise(position);
    position = octaveTransform * position;
    noise += 0.25 * valueNoise(position);
    position = octaveTransform * position;
    noise += 0.125 * valueNoise(position);
    position = octaveTransform * position;
    noise += 0.0625 * valueNoise(position);
    return noise;
}

vec3 pal(float h) {
    vec3 col = vec3(0.5) + 0.5 * cos(TAU * (vec3(0.0, 0.33, 0.67) + h));
    return mix(col, vec3(1), 0.1 * floor(h));
}

// 同名関数があるとminifyに失敗するので、sdBoxの2D版はsdRectにする
float sdRect(vec2 p, vec2 b) {
    vec2 q = abs(p) - b;
    return length(max(q, 0.)) + min(max(q.x, q.y), 0.);
}

float sdBox(vec3 p, vec3 halfSize)
{
    vec3 q = abs(p) - halfSize;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float dRope(vec3 p) {
    vec2 p1 = p.xz;

    p1 = abs(p.xz);

    float final = step(256, beat);

    vec2 q = vec2(mix(1, p.y, smoothstep(208, 210, beat)));
    q = mix(q, vec2(3), smoothstep(248, 249, beat));
    q = mix(q, vec2(1.6 - smoothstep(272, 292, beat), 1), final);

    float th = 0.0;
    float m = 1.0;

    float d = 1e7;

    for (float i = 0.; i < clamp((nBeat - 216) / 8, 2, 4); i++) {
        p1 -= m * q;
        th += p.y - beat * 0.1;
        p1 *= rot(th);
        p1 = abs(p1);
        m *= 0.4 + 0.1 * sin(p.y - beatTau / (4 + 12 * final));

        // if (i > 0) d = min(d, sdRect(p1, vec2(m)));  // 四角いツル
        if (i > 0) d = min(d, length(p1) - m);
    }

    return d * 0.5;
}

float dIFS(vec3 pos) {
    vec3 p = pos;
    vec3 offset = vec3(0.3, 0.3, 0.0);
    p -= offset;

    for (int i = 0; i < 3; i++) {
        p = abs(p + offset) - offset;
        p.xy *= rot(TAU * 0.8);
        p.zy *= rot(TAU * 0.3);
        p.xz *= rot(TAU * 0.3 * (beatPhase - 68) * (inRange(beat, 136, 160) ? 1 : 0));
    }

    return sdBox(p, vec2(mix(0.003, 0.1, smoothstep(124, 130, beat)), 0.5).xyx);
}

float dCenterObject(vec3 p)
{
    float d = 1e7;

    if (inRange(beat, 120, 300))
    {
        float ifs = dIFS(p - vec3(0, 2, 0));
        float rope = dRope(p);
        d = mix(ifs, rope, smoothstep(192, 200, beat));
    }

    return d;
}

// サイズ削減のため、120個のvec2からなるロゴ頂点をfloat配列に平坦化し、参照時にvec2へ復元する
float logoVertices[240] = float[240](
    // AEP（64頂点）
    -.53, .38, -.33, .12, -.33, -.29,
    -.26, -.37, .53, -.37, .47, -.3,
    .47, -.22, -.05, -.22, -.19, -.04,
    -.19, .22, .47, .22, .32, .38,
    -.72, .38, -1.15, -.07, -.9, -.07,
    -.71, .15, -.39, -.25, -.39, .1,
    -.6, .38, 1.16, .38, .94, .38,
    1.22, .07, 1.08, -.07, .82, -.07,
    .73, -.15, .73, -.22, 1.16, -.22,
    1.46, .07, .88, .16, .53, .16,
    .53, -.3, .62, -.4, .69, -.4,
    .69, -.01, .85, -.01, .97, .08,
    .47, -.07, .36, .09, -.13, .09,
    -.13, -.03, -.1, -.07, -.98, -.26,
    -.64, -.26, -.54, -.37, -.36, -.37,
    -.47, -.24, -.57, -.1, -.84, -.1,
    .37, .38, .53, .22, .86, .22,
    .71, .38, -1.06, -.1, -1.18, -.1,
    -1.46, -.38, -1.33, -.38, -.92, -.1,
    -1.01, -.1, -1.28, -.38, -1.18, -.38,
    .76, .38, .91, .22, 1.02, .22,
    .87, .38,

    // LTC（56頂点）
    1.11, .38, .98, .38, 1.14, .17,
    1.02, .17, 1.02, .07, .4, .07,
    .31, .17, .12, .17, .12, -.17,
    .25, -.32, .4, -.32, .4, -.18,
    .33, -.09, 1.02, -.09, 1.02, -.21,
    1.13, -.32, 1.42, -.32, 1.3, -.18,
    1.3, .13, -.69, .38, -.55, .21,
    -.2, .21, -.2, -.17, -.05, -.35,
    .01, -.35, .01, .21, .47, .21,
    .47, .29, .39, .38, -1.1, -.32,
    -.13, -.32, -.31, -.11, -.76, -.11,
    -.76, .09, -.76, .38, -.93, .38,
    -1.26, -.03, -.91, -.03, -.77, .14,
    .56, .38, .78, .11, .93, .11,
    .93, .2, .78, .38, -1.1, -.07,
    -1.2, -.07, -1.4, -.32, -1.3, -.32,
    -.94, -.07, -1.04, -.07, -1.25, -.32,
    -1.14, -.32, .85, .38, .98, .21,
    1.07, .21, .94, .38);

// 各ポリゴンのlogoVertices上の開始頂点インデックス。末尾の120は最後のポリゴンの終端
int logoOffsets[19] = int[19](
    // AEP（10ポリゴン）
    0, 12, 19, 28, 36, 41, 48, 52, 56, 60,
    // LTC（8ポリゴン）
    64, 83, 93, 98, 103, 108, 112, 116, 120
);

vec2 logoVertex(int index)
{
    index *= 2;
    return vec2(logoVertices[index], logoVertices[index + 1]);
}

float polygon(vec2 p, int first, int count)
{
    vec2 firstVertex = logoVertex(first);
    float d = dot(p - firstVertex, p - firstVertex), s = 1.;
    for (int i = first, j = first + count - 1; i < first + count; j = i, ++i) {
        vec2 vertexI = logoVertex(i), vertexJ = logoVertex(j);
        vec2 e = vertexJ - vertexI, w = p - vertexI;
        vec2 b = w - e * saturate(dot(w, e) / dot(e, e));
        d = min(d, dot(b, b));
        bvec3 c = bvec3(p.y >= vertexI.y, p.y < vertexJ.y, e.x * w.y > e.y * w.x);
        if (all(c) || all(not(c))) s = -s;
    }
    return s * sqrt(d);
}

// AEP: offset = 0, count = 10
// LTC: offset = 10, count = 8
vec3 logoPolygon(vec2 p, int lightIndex, int offset, int count)
{
    vec3 col = vec3(0);
    for (int i = offset; i < offset + count; ++i) {
        float d = polygon(p, logoOffsets[i], logoOffsets[i + 1] - logoOffsets[i]);
        float intensity = 1.0;
        if (inRange(beat, 0, 31)) intensity = step(i, (nBeat / 8) * 3 - 3 + min(int(mod(beat, 8.) * 8.), 2 + nBeat / 24));
        if (inRange(beat, 31, 32)) intensity = saturate(cos(beatTau * 8));
        if (inRange(beat, 100, 104)) intensity = saturate(sin(beatTau)) * (nBeat % 4 < 3 ? 1 : 0);
        if (inRange(beat, 256, 300)) intensity = step(floor(((nBeat - 256) / 8) * 4), i - offset + 3);
        float h = 4.5;
        if (inRange(beat, 144, 160)) h = 3 + fract(lightIndex / 7.0);
        col += intensity * pal(h) * smoothstep(fwidth(d), 0.0, d);
    }
    return col;
}

// uv.xy に模様のUV、uv.zw にパースを考慮したX/Y別の最小線幅をパックする
// XとYを単一値へまとめると、X方向だけに強いパースがかかった場合でも
// その幅がY方向へ適用され、横線が縦に太くなるため、軸ごとの幅を維持する
// MRT2のアトラス生成時は画面上の最小線幅を適用しないため、uv.zw には0を指定する
vec3 emissionPattern(vec4 uv, int lightIndex)
{
    vec3 col = vec3(0);

    // 中心が0
    vec2 p = uv.xy - 0.5;

    // 正規化
    vec2 p2 = p;
    p2.y *= 3;

    // くるくる
    float level = saturate(cos(beatTau / 2 - float(lightIndex) / 12 * TAU));

    // 各軸について、模様本来の幅0.02と画面上の1ピクセル相当幅の大きい方を採用する
    // width.xは左右の縦線、width.yは上下の横線の太さへ独立して反映される
    vec2 width = max(uv.zw, 0.02);
    float edge = smoothstep(-0.001, 0.0, sdRect(p2, vec2(0.5, 1.5) - width));
    float intensity = 0.5 + 0.5 * cos(uv.y - beatTau / 8.);
    if (inRange(beat, 184, 192)) intensity = 0.1;
    if (inRange(beat, 286, 288)) intensity = saturate(cos(beatTau * 8));
    col += pal(4.5) * edge * intensity * step(beat, 288);

    if (inRange(beat, 0, 32) || inRange(beat, 100, 160) || inRange(beat, 192, 300))
    {
        vec2 logoUv = vec2(-p2.y, p2.x);

        if (inRange(beat, 120, 208) || inRange(beat, 240, 256))
        {
            for (int i = 0; i < 2; ++i) {
                logoUv = abs(logoUv) - vec2(0.4, 1);
                logoUv *= rot(TAU * 0.25 + float(lightIndex) + (beatPhase - 68) * (inRange(beat, 136, 208) || inRange(beat, 240, 256) ? 1 : 0));
            }
        }

        int offset = 0;
        int count = 10;

        if (inRange(beat, 112, 120) || inRange(beat, 220, 256))
        {
            float b = 0.5 + 0.5 * cos(beatTau / 4 - float(lightIndex) / 2 * TAU);
            offset = int(mix(0., 10., b));
            count = int(mix(10., 8., b));
        }
        else if (lightIndex % 2 == step(beat, 192))
        {
            offset = 10;
            count = 8;
        }

        if (inRange(beat, 104, 120) && lightIndex % 4 < 2) logoUv *= rot(TAU * 0.5);

        col += logoPolygon(logoUv, lightIndex, offset, count) * (inRange(beat, 108, 112) ? level : 1);
    }

    if (inRange(beat, 32, 48))
    {
        // Sound wave pattern type1
        const int samplesPerLight = 640;
        int sampleIndex = (adjustedWaveOutPosition + lightIndex * samplesPerLight)
                        % NUM_SOUND_BUFFER_SAMPLES;
        vec2 wave = abs(waveOutSamples[sampleIndex]);
        bool isLeft = lightIndex < 6;
        float amplitude = isLeft ? wave.x : wave.y;
        const float blockCount = 16.0;
        float blockIndex = floor(uv.y * blockCount);
        vec2 blockUv = vec2(uv.x, fract(uv.y * blockCount));
        float block = step(0.02, blockUv.x) * step(blockUv.x, 0.98)
                    * step(0.08, blockUv.y) * step(blockUv.y, 0.92);
        float bar = block * step((blockIndex + 1.0) / blockCount, amplitude);
        col += bar * pal(1 + fract((isLeft ? 0.8 : 0.5) + step(8, t) * fract(blockIndex / blockCount + 0.5)));
    }

    if (inRange(beat, 48, 64))
    {
        // 単色の時間に流れるパターン
        col += level;
    }

    if (inRange(beat, 64, 80))
    {
        // ∧模様のメーター
        vec2 p3 = p2;
        float skew = abs(p3.x);
        p3.y += skew - 0.1;
        float div = 2.0;
        float pad = 1.0 / div;
        float barIndex = floor(p3.y * div) - floor((skew - 1.5) * div);
        p3.y = mod(p3.y, pad) - 0.5 * pad;
        float d = sdRect(p3, vec2(0.5, pad * 0.2));
        float barCount = div * 3.0;
        col += smoothstep(0.0, -0.001, d) * step(min(barIndex + 0.5, barCount), level * barCount) * smoothstep(76, 68, beat);
    }

    if (inRange(beat, 80, 96))
    {
        // Sound wave pattern type2
        const float barCount = 32.0;
        const int samplesPerBar = 128;
        float barIndex = floor(uv.y * barCount);
        float bar = step(0.12, fract(uv.y * barCount))
                  * step(fract(uv.y * barCount), 0.88);
        int sampleIndex = (adjustedWaveOutPosition + int(barIndex) * samplesPerBar)
                        % NUM_SOUND_BUFFER_SAMPLES;
        vec2 wave = abs(waveOutSamples[sampleIndex]);
        bool isLeft = uv.x < 0.5;
        float amplitude = isLeft ? wave.x : wave.y;
        float distanceFromCenter = 2.0 * abs(uv.x - 0.5);
        float graph = bar * step(0.03, distanceFromCenter)
                    * step(distanceFromCenter, amplitude);
        col += graph * pal((1 + fract(isLeft ? 0.8 : 0.5) + step(94, beat) * 2 * barIndex / barCount)) * (1 + step(94, beat));
    }

    if (inRange(beat, 96, 100))
    {
        col += vec3(saturate(sin(beatTau)) * (nBeat % 4 < 3 ? 1 : 0));
    }

    return 1.6 * col;
}

// タイル内の模様をライトと同じ縦横比に保ち、残りを余白にする
vec2 emissionContentScale()
{
    vec2 tileResolution = vec2(textureSize(backBuffer2, 0))
                        / vec2(EMISSION_ATLAS_COLUMNS, EMISSION_ATLAS_ROWS);
    float aspectRatio = (tileResolution.x / tileResolution.y)
                      / (LIGHT_SCALE.x / LIGHT_SCALE.y);
    return min(vec2(1.0 / aspectRatio, aspectRatio), vec2(1.0));
}

vec2 emissionAtlasUv(vec2 uv, int lightIndex, float lod)
{
    vec2 atlasSize = vec2(EMISSION_ATLAS_COLUMNS, EMISSION_ATLAS_ROWS);
    vec2 mipResolution = vec2(textureSize(backBuffer2, int(lod)));
    vec2 tile = vec2(
        float(lightIndex % EMISSION_ATLAS_COLUMNS),
        float(lightIndex / EMISSION_ATLAS_COLUMNS)
    );
    // 整数Mipごとにタイル内のテクセル中心へ制限する（非2冪の解像度にも対応）
    vec2 firstTexel = ceil(tile * mipResolution / atlasSize);
    vec2 lastTexel = floor((tile + 1.0) * mipResolution / atlasSize) - 1.0;
    uv = 0.5 + (uv - 0.5) * emissionContentScale();
    return clamp((tile + uv) * mipResolution / atlasSize,
                 firstTexel + 0.5, lastTexel + 0.5) / mipResolution;
}

struct Light {
    vec3 center;
    vec3 axisX;
    vec3 axisY;
    vec3 normal;
    vec2 halfSize;
    float halfThickness;
#if !ONE_SIDED_LIGHTS
    bool twoSided;
#endif
};

Light lights[LIGHT_COUNT];
int lightCount;

// 床上の評価点から見えるBOX表面へ、LTC用の発光矩形を厚み分だけ移動する
Light emissionFaceForPoint(Light light
#if !ONE_SIDED_LIGHTS
    , vec3 point
#endif
)
{
    float faceSide = 1.0;
#if !ONE_SIDED_LIGHTS
    if (light.twoSided)
        faceSide = dot(point - light.center, light.normal) >= 0.0 ? 1.0 : -1.0;
#endif
    light.center += faceSide * light.halfThickness * light.normal;
    return light;
}

void rectPoints(Light light, out vec3 points[4])
{
    vec3 ex = light.axisX * light.halfSize.x;
    vec3 ey = light.axisY * light.halfSize.y;
    points[0] = light.center - ex - ey;
    points[1] = light.center - ex + ey;
    points[2] = light.center + ex + ey;
    points[3] = light.center + ex - ey;
}

bool boxAxisInterval(float origin, float direction, float halfExtent, inout float nearDistance, inout float farDistance)
{
    if (abs(direction) < 1e-6)
        return abs(origin) <= halfExtent;

    float a = (-halfExtent - origin) / direction;
    float b = ( halfExtent - origin) / direction;
    nearDistance = max(nearDistance, min(a, b));
    farDistance = min(farDistance, max(a, b));
    return nearDistance <= farDistance;
}

// 発光用の矩形と同じ姿勢を持つ、厚み付きの表示用BOXとの交差
bool intersectBox(vec3 ro, vec3 rd, Light light, out float distance, out vec3 localHit, out vec3 localNormal)
{
    vec3 relativeOrigin = ro - light.center;
    vec3 localOrigin = vec3(
        dot(relativeOrigin, light.axisX),
        dot(relativeOrigin, light.axisY),
        dot(relativeOrigin, light.normal)
    );
    vec3 localDirection = vec3(
        dot(rd, light.axisX),
        dot(rd, light.axisY),
        dot(rd, light.normal)
    );
    vec3 halfSize = vec3(light.halfSize, light.halfThickness);

    float nearDistance = -1000000.0;
    float farDistance = 1000000.0;
    if (!boxAxisInterval(localOrigin.x, localDirection.x, halfSize.x, nearDistance, farDistance)
        || !boxAxisInterval(localOrigin.y, localDirection.y, halfSize.y, nearDistance, farDistance)
        || !boxAxisInterval(localOrigin.z, localDirection.z, halfSize.z, nearDistance, farDistance)
        || farDistance <= 0.0)
        return false;

    distance = nearDistance > 0.0 ? nearDistance : farDistance;
    localHit = localOrigin + distance * localDirection;
    vec3 faceDistance = abs(localHit / halfSize);
    if (faceDistance.x > faceDistance.y && faceDistance.x > faceDistance.z)
        localNormal = vec3(sign(localHit.x), 0.0, 0.0);
    else if (faceDistance.y > faceDistance.z)
        localNormal = vec3(0.0, sign(localHit.y), 0.0);
    else
        localNormal = vec3(0.0, 0.0, sign(localHit.z));
    return true;
}

// カメラレイを指定した発光面へ投影し、交点をライトのUVへ変換する
// closestLocalHitの微分を直接使うと、隣接ピクセルが別のライトへ当たる境界で
// 異なる面の座標が混ざるため、線幅を過大・過小評価してしまう
// そこで中心レイと同じlight、同じfaceZへ隣接レイも投影し、同一平面上のUV差を求める
vec2 emissionUvForRay(vec3 ro, vec3 rd, Light light, float faceZ)
{
    // レイの原点と方向を、ライトのaxisX、axisY、normalを基底とするローカル座標へ変換する
    vec3 relativeOrigin = ro - light.center;
    vec3 localOrigin = vec3(
        dot(relativeOrigin, light.axisX),
        dot(relativeOrigin, light.axisY),
        dot(relativeOrigin, light.normal)
    );
    vec3 localDirection = vec3(
        dot(rd, light.axisX),
        dot(rd, light.axisY),
        dot(rd, light.normal)
    );
    // ローカルZが発光面の位置faceZになるレイ上の距離を求める
    float distance = (faceZ - localOrigin.z) / localDirection.z;

    // 交点のローカルXYを、ライト左下が0、右上が1になるUVへ変換する
    return 0.5 + 0.5 * (localOrigin.xy + distance * localDirection.xy) / light.halfSize;
}

vec3 integrateEdgeVec(vec3 v1, vec3 v2)
{
    float x = dot(v1, v2);
    float y = abs(x);
    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;
    float thetaSinTheta = x > 0.0
        ? v
        : 0.5 * inversesqrt(max(1.0 - x * x, 0.0000001)) - v;
    return cross(v1, v2) * thetaSinTheta;
}

void clipQuadToHorizon(inout vec3 L[5], out int n)
{
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;
    n = 0;

    if (config == 1) {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    } else if (config == 2) {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    } else if (config == 3) {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    } else if (config == 4) {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    } else if (config == 6) {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    } else if (config == 7) {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    } else if (config == 8) {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = L[3];
    } else if (config == 9) {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    } else if (config == 11) {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    } else if (config == 12) {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    } else if (config == 13) {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    } else if (config == 14) {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    } else if (config == 15) {
        n = 4;
    }

    if (n == 3) L[3] = L[0];
    if (n == 4) L[4] = L[0];
}

vec3 fetchFilteredEmission(vec3 p1, vec3 p2, vec3 p4, vec3 direction, int lightIndex)
{
    vec3 edge1 = p2 - p1;
    vec3 edge2 = p4 - p1;
    vec3 planeNormal = cross(edge1, edge2);
    float planeAreaSquared = dot(planeNormal, planeNormal);
    float denominator = dot(planeNormal, direction);
    if (abs(denominator) < 0.000001 || planeAreaSquared < 0.00000001)
        return textureLod(backBuffer2, emissionAtlasUv(vec2(0.5), lightIndex, 0.0), 0.0).rgb;

    float planeDistance = dot(planeNormal, p1) / denominator;
    vec3 planePoint = planeDistance * direction - p1;
    float edge11 = dot(edge1, edge1);
    float edge12 = dot(edge1, edge2);
    vec3 edge2Ortho = edge2 - edge1 * edge12 / edge11;
    vec2 uv;
    uv.y = dot(edge2Ortho, planePoint) / dot(edge2Ortho, edge2Ortho);
    uv.x = dot(edge1, planePoint) / edge11 - edge12 / edge11 * uv.y;

    float distanceToLight = abs(planeDistance) / pow(planeAreaSquared, 0.25);
    vec2 tileResolution = vec2(textureSize(backBuffer2, 0))
                        / vec2(EMISSION_ATLAS_COLUMNS, EMISSION_ATLAS_ROWS);
    vec2 contentResolution = tileResolution * emissionContentScale();
    // distanceToLightは面積の平方根で正規化しているため、画素数も面積から求める
    float textureResolution = sqrt(contentResolution.x * contentResolution.y);
    // タイルの内側に有効なテクセルを確保できるMipまでに制限する
    float maxLod = floor(log2(max(min(contentResolution.x, contentResolution.y) * 0.5, 1.0)));
    float lod = clamp(log2(max(textureResolution * distanceToLight, 1.0)), 0.0, maxLod);
    // 小数LODでUVを動かすと、明るい枠を繰り返し横切って接地部に縞が出る
    // 各Mipを固定したUVで読み、色だけを補間する
    float lod0 = floor(lod);
    float lod1 = min(lod0 + 1.0, maxLod);
    vec3 a = textureLod(backBuffer2, emissionAtlasUv(uv, lightIndex, lod0), lod0).rgb;
    vec3 b = textureLod(backBuffer2, emissionAtlasUv(uv, lightIndex, lod1), lod1).rgb;
    return mix(a, b, fract(lod));
}

vec3 ltcEvaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4]
#if !ONE_SIDED_LIGHTS
    , bool twoSided
#endif
    , int lightIndex)
{
    vec3 projected = V - N * dot(V, N);
    vec3 T1 = dot(projected, projected) > 1e-8
        ? normalize(projected)
        : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(N, T1);
    Minv *= transpose(mat3(T1, T2, N));

    vec3 L[5];
    L[0] = Minv * (points[0] - P);
    L[1] = Minv * (points[1] - P);
    L[2] = Minv * (points[2] - P);
    L[3] = Minv * (points[3] - P);

    vec3 originalL0 = L[0];
    vec3 originalL1 = L[1];
    vec3 originalL3 = L[3];

    int n;
    clipQuadToHorizon(L, n);
    if (n == 0)
        return vec3(0.0);

    for (int i = 0; i < 5; ++i)
        L[i] = normalize(L[i]);

    vec3 vectorSum = integrateEdgeVec(L[0], L[1])
                   + integrateEdgeVec(L[1], L[2])
                   + integrateEdgeVec(L[2], L[3]);
    if (n >= 4) vectorSum += integrateEdgeVec(L[3], L[4]);
    if (n == 5) vectorSum += integrateEdgeVec(L[4], L[0]);
#if ONE_SIDED_LIGHTS
    float sum = max(0.0, vectorSum.z);
#else
    float sum = twoSided ? abs(vectorSum.z) : max(0.0, vectorSum.z);
#endif
    if (sum <= 0.0 || dot(vectorSum, vectorSum) < 0.00000001)
        return vec3(0.0);

    // rectPoints reverses the LTC winding so +normal is emissive. Keep texture X/Y unchanged.
    vec3 emission = fetchFilteredEmission(originalL0, originalL3, originalL1, normalize(vectorSum), lightIndex);
    return sum * emission;
}

float map(vec3 p)
{
    // plane
    float d = p.y;

    // lights
    for (int i = 0; i < lightCount; ++i) {
        Light light = lights[i];
        vec3 relativePosition = p - light.center;
        vec3 localPosition = vec3(
            dot(relativePosition, light.axisX),
            dot(relativePosition, light.axisY),
            dot(relativePosition, light.normal)
        );
        d = min(d, sdBox(
            localPosition,
            vec3(light.halfSize, light.halfThickness)
        ));
    }

    // CenterObject
    return min(d, dCenterObject(p));
}

vec3 calcNormal(vec3 p, float eps) {
    vec2 k = vec2(eps, 0);
    return normalize(vec3(map(p + k.xyy).x - map(p - k.xyy).x, map(p + k.yxy).x - map(p - k.yxy).x, map(p + k.yyx).x - map(p - k.yyx).x));
}

// https://iquilezles.org/articles/rmshadows
float calcSoftShadow(vec3 ro, vec3 rd, float mint, float tmax)
{
    float visibility = 1.0;
    float t = mint;
    for (int i = 0; i < 24; ++i) {
        float h = map(ro + rd * t);
        visibility = min(visibility, saturate(8.0 * h / t));
        t += clamp(h, 0.01, 0.2);
        if (visibility < 0.004 || t > tmax)
            break;
    }
    visibility = saturate(visibility);
    return visibility * visibility * (3.0 - 2.0 * visibility);
}

// https://iquilezles.org/articles/nvscene2008/rwwtt.pdf
float calcAO(vec3 P, vec3 N)
{
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 1; i < 10; i++) {
        float h = 0.1 * i;
        float d = map(P + h * N);
        occ += (h - d) * sca;
        sca *= 0.8;
        // if (occ > 0.35) break;
    }

    return saturate(1.0 - occ);
}

float lightVisibility(vec3 P, vec3 N, float minimumVisibility)
{
    vec3 shadowOrigin = P + 0.01 * N;
    vec3 lightPosition = vec3(0.0, 6, 0.0);
    vec3 toLight = lightPosition - shadowOrigin;
    float lightDistance = length(toLight);
    float shadow = calcSoftShadow(
        shadowOrigin,
        toLight / lightDistance,
        0.01,
        lightDistance - 0.002
    );
    float occlusion = calcAO(P, N);
    // return occlusion;  // debug
    return mix(minimumVisibility * occlusion, 1.0, shadow);
}

#if METALLIC_ONE
vec3 surfaceLighting(vec3 P, vec3 N, vec3 V, vec3 baseColor, float roughness, float minimumVisibility)
{
    float ndotv = saturate(dot(N, V));
    vec2 lutUv = vec2(roughness, sqrt(1.0 - ndotv));
    vec4 params = textureLod(backBuffer1, referenceLUTUv(lutUv, true), 0.0);
    float normalization = textureLod(backBuffer1, referenceLUTUv(lutUv, false), 0.0).x;
    mat3 Minv = mat3(
        vec3(1.0, 0.0, params.y),
        vec3(0.0, params.z, 0.0),
        vec3(params.w, 0.0, params.x)
    );

    vec3 specular = vec3(0.0);
    float visibility = lightVisibility(P, N, minimumVisibility);
    for (int i = 0; i < lightCount; ++i) {
        Light light = lights[i];
        Light emissionFace = emissionFaceForPoint(light
#if !ONE_SIDED_LIGHTS
            , P
#endif
        );
        vec3 points[4];
        rectPoints(emissionFace, points);
        vec3 specularLtc = ltcEvaluate(N, V, P, Minv, points
#if !ONE_SIDED_LIGHTS
            , light.twoSided
#endif
            , i);
        specular += specularLtc * baseColor * normalization;
    }

    return visibility * specular;
}
#else
vec3 surfaceLighting(vec3 P, vec3 N, vec3 V, vec3 baseColor, float roughness, float metallic, float minimumVisibility)
{
    vec3 specularColor = mix(vec3(0.04), baseColor, metallic);
    vec3 diffuseReflectance = baseColor * (1.0 - metallic);
    float ndotv = saturate(dot(N, V));
    vec2 lutUv = vec2(roughness, sqrt(1.0 - ndotv));
    vec4 params = textureLod(backBuffer1, referenceLUTUv(lutUv, true), 0.0);
    float normalization = textureLod(backBuffer1, referenceLUTUv(lutUv, false), 0.0).x;
    mat3 Minv = mat3(
        vec3(1.0, 0.0, params.y),
        vec3(0.0, params.z, 0.0),
        vec3(params.w, 0.0, params.x)
    );

    vec3 specular = vec3(0.0);
    vec3 diffuse = vec3(0.0);
    float visibility = lightVisibility(P, N, minimumVisibility);
    for (int i = 0; i < lightCount; ++i) {
        Light light = lights[i];
        Light emissionFace = emissionFaceForPoint(light
#if !ONE_SIDED_LIGHTS
            , P
#endif
        );
        vec3 points[4];
        rectPoints(emissionFace, points);
        vec3 specularLtc = ltcEvaluate(N, V, P, Minv, points
#if !ONE_SIDED_LIGHTS
            , light.twoSided
#endif
            , i);
        specular += specularLtc * specularColor * normalization;
        diffuse += ltcEvaluate(N, V, P, mat3(1.0), points
#if !ONE_SIDED_LIGHTS
            , light.twoSided
#endif
            , i)
                 * diffuseReflectance;
    }

    return visibility * (specular + diffuse);
}
#endif

float luminance(vec3 color)
{
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 bloomSample(vec4 color)
{
    // The mip already averages the bright RGB contribution. Multiplying by the
    // averaged mask again would attenuate small emitters twice at high LODs.
    return color.rgb * (1.0 - exp(-80.0 * color.a));
}

vec3 bloomLevel(sampler2D image, vec2 uv, float lod)
{
    vec2 offset = exp2(lod) / vec2(textureSize(image, 0));
    vec2 edge = 0.5 * offset;
    vec3 result = vec3(0.0);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float weight = float((x == 0 ? 2 : 1) * (y == 0 ? 2 : 1)) / 16.0;
            vec2 sampleUv = clamp(uv + offset * vec2(x, y), edge, 1.0 - edge);
            result += weight * bloomSample(textureLod(image, sampleUv, lod));
        }
    }
    return result;
}

vec3 bloom(sampler2D image, vec2 uv)
{
    return BLOOM_INTENSITY * (
          0.35 * bloomLevel(image, uv, 2.5)
        + 0.30 * bloomLevel(image, uv, 3.5)
        + 0.22 * bloomLevel(image, uv, 4.5)
        + 0.13 * bloomLevel(image, uv, 5.5)
    );
}

vec3 postprocess(sampler2D image, vec2 uv)
{
    vec2 texelSize = 1.0 / vec2(textureSize(image, 0));
    vec3 rgbNW = textureLod(image, uv + texelSize * vec2(-1.0,  1.0), 0.0).rgb;
    vec3 rgbNE = textureLod(image, uv + texelSize * vec2( 1.0,  1.0), 0.0).rgb;
    vec3 rgbSW = textureLod(image, uv + texelSize * vec2(-1.0, -1.0), 0.0).rgb;
    vec3 rgbSE = textureLod(image, uv + texelSize * vec2( 1.0, -1.0), 0.0).rgb;
    vec3 rgbM = textureLod(image, uv, 0.0).rgb;

    float lumaNW = luminance(rgbNW);
    float lumaNE = luminance(rgbNE);
    float lumaSW = luminance(rgbSW);
    float lumaSE = luminance(rgbSE);
    float lumaM = luminance(rgbM);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 direction = vec2(
        -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
         (lumaNW + lumaSW) - (lumaNE + lumaSE)
    );
    float directionReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * 0.125),
        1.0 / 128.0
    );
    float inverseDirectionMin = 1.0
        / (min(abs(direction.x), abs(direction.y)) + directionReduce);
    direction = clamp(direction * inverseDirectionMin, vec2(-8.0), vec2(8.0))
              * texelSize;

    vec3 rgbA = 0.5 * (
        textureLod(image, uv + direction * (1.0 / 3.0 - 0.5), 0.0).rgb
      + textureLod(image, uv + direction * (2.0 / 3.0 - 0.5), 0.0).rgb
    );
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        textureLod(image, uv + direction * -0.5, 0.0).rgb
      + textureLod(image, uv + direction *  0.5, 0.0).rgb
    );
    float lumaB = luminance(rgbB);
    vec3 antialiased = lumaB < lumaMin || lumaB > lumaMax ? rgbA : rgbB;

    // bloom
    vec3 glow = saturate(bloom(image, uv));

    // AA適用
    vec3 color = antialiased + glow * (1.0 - antialiased);

    // トーンマップ
    // color = color / (1.0 + color);

    // ガンマ補正
    color = pow(max(color, 0.0), vec3(1.0 / 2.2));

    // fadeout
    // color *= smoothstep(290, 288, beat);

    return color;
}

vec3 render(vec3 ro, vec3 rd, vec3 rdRight, vec3 rdUp)
{
    bool hit = false;
    vec3 color = vec3(0);
    vec3 baseColor;
    float roughness;
#if !METALLIC_ONE
    float metallic = 1.0;
#endif
    vec3 P, N;
    float minimumVisibility = 0.05;

    // シーン中の交差点の最短距離
    float tmin = 1e7;

    // 床との衝突判定
    // カメラが床の上にあるという前提でrd.yだけで判定
    if (rd.y < -1e-6) {
        tmin = -ro.y / rd.y;
        P = ro + rd * tmin;
        vec4 floorNoise = 0.5 + 0.5 * fbm(P.xz * 4.0);
        // vec2 floorSlope = floorNoise.rg * 2.0 - 1.0;
        // N = vec3(0, 1, 0);

        // grid cell
        vec2 gridDistance = abs(fract(P.xz + 0.5) - 0.5);
        float grid = min(gridDistance.x, gridDistance.y);

        roughness = mix(0.04, 0.5, floorNoise.b);
        roughness = mix(0.7, roughness, smoothstep(0.015, 0.04, grid));
        // roughness = 0.3;

        baseColor = vec3(0.4) * mix(0.88, 1.0, floorNoise.a);
        // metallic = 1.0;
        hit = true;
    }

    // Lightsとの衝突判定
    int closestLightIndex = -1;
    vec3 closestLocalHit = vec3(0.0);
    vec3 closestLocalNormal = vec3(0.0);
    float tminLight = 1e7;
    for (int i = 0; i < lightCount; ++i) {
        Light light = lights[i];
        float lightDistance;
        vec3 localHit;
        vec3 localNormal;
        if (intersectBox(ro, rd, light, lightDistance, localHit, localNormal) && lightDistance < tmin) {
            tmin = tminLight = lightDistance;
            closestLightIndex = i;
            closestLocalHit = localHit;
            closestLocalNormal = localNormal;
            hit = true;
        }
    }

    // CenterObjectとの衝突判定
    vec3 p;
    float t = 0.;
    float eps = 0.001;

    for (int i = 0; i < 100; i++) {
        p = ro + rd * t;
        float d = dCenterObject(p);
        t += d;
        eps = t * 0.0001;
        if (d < eps || t > tmin) {
            if (t < tmin)
            {
                hit = true;
                tmin = t;
                baseColor = vec3(0.1);
                roughness = 0.2;
                // metallic = 1.0;
            }
            break;
        }
    }

    // mainで解析的に生成した右・上の隣接ピクセルレイを使う
    if (closestLightIndex >= 0 && tmin == tminLight) {
        Light light = lights[closestLightIndex];
#if ONE_SIDED_LIGHTS
        bool hitEmissionFace = closestLocalNormal.z > 0.5;
#else
        bool hitEmissionFace = light.twoSided
            ? abs(closestLocalNormal.z) > 0.5
            : closestLocalNormal.z > 0.5;
#endif
        if (hitEmissionFace) {
            vec2 lightUv = 0.5 + 0.5 * closestLocalHit.xy / light.halfSize;
            float faceZ = closestLocalNormal.z * light.halfThickness;

            // 画面上で1px未満にならないように補正したwidthを求める処理
            // X/Yの隣接レイを中心レイと同じ発光面へ投影し、UV差を成分ごとに比較する
            // 各UV軸で大きい方を採用することで、どちらの画面方向へ傾いた場合も1ピクセル未満にならない保守的な最小幅を得る
            // emissionPattern内でYを3倍しているため、UV上の1ピクセル相当幅を模様の座標系へ合わせて vec2(1, 3) とする
            vec2 emissionPixelWidth = max(
                abs(emissionUvForRay(ro, rdRight, light, faceZ) - lightUv),
                abs(emissionUvForRay(ro, rdUp, light, faceZ) - lightUv)
            ) * vec2(1, 3);

            color += emissionPattern(vec4(lightUv, emissionPixelWidth), closestLightIndex);
        }

        minimumVisibility = 0.0;

        // BOXの厚み部分は、発光面を囲む暗い筐体として表示する
        baseColor = vec3(0.1);
        roughness = 0.2;

        // 解析的に法線を求める版
        // N = normalize(
        //         closestLocalNormal.x * light.axisX
        //     + closestLocalNormal.y * light.axisY
        //     + closestLocalNormal.z * light.normal
        // );
    }

    if (hit)
    {
        P = ro + tmin * rd;
        N = calcNormal(P, eps);
#if METALLIC_ONE
        color += 2.0 * surfaceLighting(P, N, -rd, baseColor, roughness, minimumVisibility);
#else
        color += 2.0 * surfaceLighting(P, N, -rd, baseColor, roughness, metallic, minimumVisibility);
#endif
    }

    return color;
}

// t          : 0.0 ～ 1.0
// majorRadius: トーラス全体の半径
// minorRadius: チューブ方向の半径
// p, q       : Torus Knot の巻き数（互いに素が推奨）
//
// 例:
//   (p, q) = (2, 3) -> Trefoil Knot

vec3 torusKnot(
    float t,
    float majorRadius,
    float minorRadius,
    float p,
    float q)
{
    float u = t * TAU;

    float pu = p * u;
    float qu = q * u;

    float r = majorRadius + minorRadius * cos(qu);

    return vec3(
        r * cos(pu),
        minorRadius * sin(qu),
        r * sin(pu)
    );
}

vec3 torusKnotTangent(
    float t,
    float majorRadius,
    float minorRadius,
    float p,
    float q)
{
    float u = t * TAU;

    float pu = p * u;
    float qu = q * u;

    float r  = majorRadius + minorRadius * cos(qu);
    float dr = -minorRadius * q * sin(qu);

    vec3 tangent = vec3(
        dr * cos(pu) - r * p * sin(pu),
        minorRadius * q * cos(qu),
        dr * sin(pu) + r * p * cos(pu)
    );

    return normalize(tangent);
}

void initBeat(float offset)
{
    beat = (time + offset) * 140.0 / 60.0;
    beatTau = beat * TAU;
    beatPhase = phase(beat / 2.);
    nBeat = int(beat);
    adjustedWaveOutPosition = waveOutPosition + int(offset * SAMPLES_PER_SECOND);
}

void main()
{
    initBeat(1.0 / 60.0);

    ivec2 pixel = ivec2(gl_FragCoord.xy);
    bool cacheReady = all(lessThan(
        abs(texelFetch(backBuffer1, ivec2(0), 0) - vec4(1.0)),
        vec4(0.000001)
    ));
    vec2 screenUv = gl_FragCoord.xy / resolution;
    vec2 p = (2.0 * gl_FragCoord.xy - resolution) / resolution.y;

    //=====================================================
    // Timeline
    //=====================================================
    float fov = 45.0;
    vec3 worldUp = vec3(0, 1, 0);

    lightCount = 12;

#if !defined(EXPORT_EXECUTABLE) && 0
    lightCount = debugLightCount;
#endif

    // ランダムカメラ
    vec4 hash = hash42(vec2(nBeat / 4));
    float camY, radius, angle;
    vec3 ro, target;

    if ((nBeat / 4) % 2 == 0)
    {
        // 近いカメラ
        camY = mix(1, 4, hash.x);
        radius = mix(2, 6, hash.y);
        angle = beatTau / 64.0 + TAU * hash.z;
    }
    else
    {
        // 遠いカメラ
        camY = mix(2, 6, hash.z);
        radius = mix(7, 12, hash.y);
        angle = beatTau / 64.0 + TAU * hash.x;
    }

    ro = vec3(sin(angle) * radius, camY, cos(angle) * radius);
    target = vec3(0.0, 1.5, 0.0);

    if (inRange(beat, 0, 32))
    {
        lightCount = 1;
        ro = mix(vec3(0.0, 1.0, 3.0), vec3(0.0, 1.5, 6.0), blend);
        target = mix(vec3(0.0, 0.5, 0.0), vec3(0.0, 0.8, 0.0), blend);
    }
    else if (inRange(beat, 32, 40))
    {
        ro = vec3(0, 2, 7 - t * 0.1);
        target = ro + vec3(0, -0.3, -1);
    }
    else if (inRange(beat, 60, 96))
    {
        radius = 10 + 0.1 * t;
        angle = beatTau / 64.0;
        ro = vec3(sin(angle) * radius, 3.2, cos(angle) * radius);
    }
    else if (inRange(beat, 120, 128))
    {
        radius = 3;
        angle = beatTau / 64.0;
        ro = vec3(sin(angle) * radius, 1, cos(angle) * radius);
        target = vec3(0, 2, 0);
        fov = 60;
    }
    else if (inRange(beat, 128, 144))
    {
        float blend2 = smoothstep(0.0, 0.5, blend);
        ro = mix(vec3(0, 1.5, 3), vec3(0, 2, 3.5), blend2);
        target = vec3(0, 2, 0);
        fov = 60;
    }
    else if (inRange(beat, 144, 148))
    {
        radius = 6;
        angle = beatTau / 64.0;
        ro = vec3(sin(angle) * radius, 5, cos(angle) * radius);
    }
    else if (inRange(beat, 216, 224))
    {
        ro = vec3(0.0, 1.5, -2.0);
        target = vec3(-1.0, 1.5, -4.0);
    }
    else if (inRange(beat, 256, 300))
    {
        lightCount = 1;
        ro = mix(vec3(0.0, 1.5, 6.0), vec3(0.0, 1.5, 10.0), blend);
        target = vec3(0.0, 1.0, 0.0);
    }

    const float COLUMN_X = 2.4;
    const float COLUMN_SPACING = 1.75;
    const float CIRCLE_RADIUS = 4.0;

    for (int i = 0; i < lightCount; ++i) {
        vec3 rotation = vec3(0.0);
        vec3 position = vec3(0.0, LIGHT_HEIGHT, 0.0);

        if (inRange(beat, 32, 208))
        {
            // float circleBlend = easeInOutCubic(remapFrom(beat, 64, 80));
            float circleBlend = smoothstep(64, 80, beat);
            float linearBlend = 0.0;

            bool leftColumn = i < LIGHT_COUNT / 2;
            int row = leftColumn ? LIGHT_COUNT / 2 - 1 - i : i - LIGHT_COUNT / 2;
            vec3 columnPosition = vec3(
                leftColumn ? -COLUMN_X : COLUMN_X,
                LIGHT_HEIGHT,
                (float(row) - 2.5) * COLUMN_SPACING
            );
            float columnRotationY = leftColumn ? 0.25 * TAU : 0.75 * TAU;

            float circleAngle = TAU * float(i) / float(LIGHT_COUNT) + TAU * 0.25;
            circleAngle = mod(circleAngle, TAU);
            vec3 circlePosition = vec3(
                cos(circleAngle) * CIRCLE_RADIUS,
                LIGHT_HEIGHT,
                sin(circleAngle) * CIRCLE_RADIUS
            );
            float circleRotationY = 0.75 * TAU - circleAngle;

            position = mix(columnPosition, circlePosition, circleBlend);
            rotation = vec3(0.0, mix(columnRotationY, circleRotationY, circleBlend), 0.0);
        }
        else if (inRange(beat, 208, 224))
        {
            float circleAngle = TAU * (float(i) / float(LIGHT_COUNT) + beatPhase / 16);
            float circleRadius = 8.0;
            position = vec3(
                cos(circleAngle) * circleRadius,
                LIGHT_HEIGHT + smoothstep(212, 214, beat) * sin(6 * circleAngle),
                sin(circleAngle) * circleRadius
            );

            rotation = vec3(TAU * 0.75 - circleAngle, 0.0, TAU * 0.25);
        }
        else if (inRange(beat, 224, 256))
        {
            // Torus Knot の上に配置する
            float t = fract(float(i) / float(lightCount) + beat / 64.0);
            float majorRadius = 3.0;
            float minorRadius = 1.0;
            float p = 2.0;
            float q = 5.0;
            position = 1.5 * torusKnot(t, majorRadius, minorRadius, p, q);
            position.y += 2 * LIGHT_HEIGHT;

            // ローカル +Z を tangent の方向へ向ける
            vec3 tangent = torusKnotTangent(t, majorRadius, minorRadius, p, q);
            rotation = vec3(
                -asin(clamp(tangent.y, -1.0, 1.0)),
                atan(tangent.x, tangent.z),
                0.0
            );
        }

        // ローカル +Z 面が発光方向。rotation は XYZ オイラー角（ラジアン）
        mat3 rotationMatrix = rotationXYZ(rotation);
        vec3 axisX = rotationMatrix * vec3(1.0, 0.0, 0.0);
        vec3 axisY = rotationMatrix * vec3(0.0, 1.0, 0.0);
        lights[i] = Light(
            position,
            axisX,
            axisY,
            cross(axisX, axisY),
            LIGHT_SCALE.xy,
            LIGHT_SCALE.z
#if !ONE_SIDED_LIGHTS
            , false
#endif
        );
    }

    // 右手系でレイを生成（カメラのfwd方向がZ-）
    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(fwd, worldUp));
    vec3 up = normalize(cross(right, fwd));
    vec3 focal = fwd / tan(fov * TAU / 720.);
    vec3 ray = right * p.x + up * p.y + focal;
    vec3 rd = normalize(ray);
    float pixelStep = 2.0 / resolution.y;
    vec3 rdRight = normalize(ray + pixelStep * right);
    vec3 rdUp = normalize(ray + pixelStep * up);

#if !defined(EXPORT_EXECUTABLE) && 0
    // debug camera
    ro = cameraInWorld[3].xyz;
    rd = mat3(cameraInWorld) * normalize(vec3(p * tanFovY, -1));
#endif

    //=====================================================
    // MRT用の出力処理
    //=====================================================
    if (cacheReady) {
        outColor0 = vec4(postprocess(backBuffer3, screenUv), 1.0);
        vec3 scene = render(ro, rd, rdRight, rdUp);
        float bloomMask = smoothstep(0.70, 0.95, luminance(scene));
        outColor3 = vec4(scene, bloomMask);
        // outColor0 = vec4((0.5 + 0.5 * fbm(screenUv * 30.0)).rrr, 1);  // debug
        // outColor0 = texture(backBuffer2, screenUv);  // debug
        // outColor0 = pal(screenUv.x).rgbr;  // debug
        // outColor0 = vec4(scene, 1);  // debug
        outColor1 = texelFetch(backBuffer1, pixel, 0);
    } else {
        outColor0 = vec4(0.0, 0.0, 0.0, 1.0);
        outColor3 = vec4(0.0, 0.0, 0.0, 1.0);
        outColor1 = backBuffer1Texel(pixel);
    }

    vec2 atlasSize = vec2(EMISSION_ATLAS_COLUMNS, EMISSION_ATLAS_ROWS);
    ivec2 atlasTile = ivec2(min(floor(screenUv * atlasSize), atlasSize - 1.0));
    int atlasLightIndex = atlasTile.y * EMISSION_ATLAS_COLUMNS + atlasTile.x;
    vec2 atlasUv = (fract(screenUv * atlasSize) - 0.5) / emissionContentScale() + 0.5;

    // 余白は端の色で埋め、Mip生成時に模様の境界が黒くならないようにする
    atlasUv = saturate(atlasUv);
    initBeat(2.0 / 60.0);
    outColor2 = vec4(emissionPattern(vec4(atlasUv, 0, 0), atlasLightIndex), 1.0);
}
