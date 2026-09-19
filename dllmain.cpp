/*
 * OpenYSM.cpp — aligned rebuild against the shipped libysm-core.so binary.
 *
 * Base: upstream OpenYSMDev/openysm.cpp (MIT, Meow404club fork pin 3e86bb0).
 * Deviations from upstream are backed by objdump evidence of the shipped
 * binary in the ModernYSM working area (tmp/native-align/GM_*.asm):
 *
 *   [A] nInitSIMD — new export (11th symbol). Caches jfieldID/jmethodID
 *       handles of the live BufferBuilder subclass so the render hot path can
 *       write vertices straight into the builder's direct ByteBuffer.
 *       Evidence: GM_nInitSIMD.asm:1-332 (strings decoded from .rodata
 *       0xa46a/0xa9c3/0xa95f/0xa7e9/0xa67c/0xa036/0xa02e/0xa975).
 *   [B] nComputeModelVertices — 5th parameter jfloatArray stateArray inserted
 *       after animArray (descriptor "(JLjava/lang/Object;[F[F[FIIIFFFF)V").
 *       When any bone carries animData[bone*12+11] == 1.0f, the native side
 *       fills stateArray[bone*4 .. bone*4+2] with parentGlobal * T applied to
 *       (-16, 16, 16). Evidence: GM_nComputeModelVertices.asm scan
 *       16649-16673 (sentinel, constant 9c50 = 1.0), write 170d5-171c2
 *       (constants 9c40 = {-16, 16}, 9c58 = 16, bounds 170fb), release
 *       mode-0 copy-back 19ea0-19ebd.
 *   [C] translucent byte — nInitModelCache READS the per-quad translucent
 *       byte written by GeoModel.buildNativeCache() and partitions quads into
 *       four buckets by (cullable, translucent). Evidence:
 *       GM_nInitModelCache.asm:375 (movzbl at quad base), :436-469 (four-way
 *       push targets 15db8/15de0; cullable steers between two list pairs).
 *   [D] Subtree skip flag still sourced from animData[bone*12+10] with the
 *       upstream polarity (!= 0.0f skips the subtree). The upstream
 *       "scale == 0 -> skip subtree" early-out is NOT present in the shipped
 *       binary. Evidence: GM_nComputeModelVertices.asm 16b04 (load +0x28),
 *       16a53-16a6f / 1765d-1767a (k-advance decision).
 */

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3,unroll-loops")
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx2,fma,bmi2")
#endif
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
#define SSE2NEON_SUPPRESS_WARNINGS
#include "sse2neon.h"
#else
#include <immintrin.h>
#endif

#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <memory>
#include "jni.h"
#include <cstdint>

#if defined(__GLIBC__) || defined(__BIONIC__)
#define FAST_SINCOS(x, s, c) sincosf((x), (s), (c))
#else
inline void ms_sincosf(float x, float *s, float *c) {
    *s = std::sin(x);
    *c = std::cos(x);
}

#define FAST_SINCOS(x, s, c) ms_sincosf((x), (s), (c))
#endif

#if defined(__FMA__)
#define MADD_PS(a, b, c) _mm_fmadd_ps((a), (b), (c))
#else
#define MADD_PS(a, b, c) _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#endif

static jclass g_NativeModelRendererClass = nullptr;
static jmethodID g_submitVerticesID = nullptr;

// [A] Cached handles populated by nInitSIMD (GM_nInitSIMD.asm:168-274;
// bss slots of the shipped binary noted per field).
static jclass g_BufferBuilderClass = nullptr;          // bss 0x449f8
static jclass g_SodiumBufferBuilderClass = nullptr;    // bss 0x449e8 (optional)
static jfieldID g_sodiumBuilderFieldID = nullptr;      // bss 0x449f0 ("builder")
static jmethodID g_ensureCapacityMethodID = nullptr;   // bss 0x449e0 ("(I)V")
static jfieldID g_bufferFieldID = nullptr;             // bss 0x44a10 (ByteBuffer)
static jfieldID g_verticesFieldID = nullptr;           // bss 0x44a00 ("I")
static jfieldID g_nextElementByteFieldID = nullptr;    // bss 0x44a08 ("I")
static jfieldID g_modeFieldID = nullptr;               // bss 0x44a28 (cached;
                                                       // never read back by the
                                                       // shipped binary)

// [A2] 26.2/26.3 staging surface (native-262). 26.2 rewrote BufferBuilder's
// field face: `buffer` became a ByteBufferBuilder (raw native-memory staging),
// `nextElementByte`/`ensureCapacity` are gone (the write position now lives
// inside ByteBufferBuilder.reserve), `mode` became primitiveTopology. neoform
// 26.3 sources diff field-identical, so one staging path covers both. The
// mapping NAMES still arrive via the BufferBuilderMixin annotations; the
// SURFACE is probed at runtime instead of assumed:
//   legacy : buffer Ljava/nio/ByteBuffer; + nextElementByte I + ensureCapacity (I)V
//   staging: buffer Lcom/mojang/blaze3d/vertex/ByteBufferBuilder; + vertices I
//            + ByteBufferBuilder.reserve(I)J (resolved lazily off the live
//            buffer object — no class name hardcode beyond the field probe)
static bool g_stagingBufferField = false;              // buffer probe hit the
                                                       // ByteBufferBuilder desc
static jfieldID g_vertexSizeFieldID = nullptr;         // "vertexSize" I — 36B
                                                       // record guard (staging
                                                       // only; optional)
static jmethodID g_reserveMethodID = nullptr;          // ByteBufferBuilder.reserve(I)J
static bool g_reserveProbeDead = false;                // lookup failed; stop retrying

struct FastQuad {
    int boneIdx;
    bool cullable;
    __m128 x, y, z;
    __m128 u, v;
    float nx, ny, nz;
};

struct NativeBone {
    int parentIdx;
    int partMask;
    bool glow;
    float pivotX, pivotY, pivotZ;
    int quadStart;
    int quadCount;
    int subtreeCount;
    float aabbMin[3];
    float aabbMax[3];
};

struct alignas(16) PrecomputedBoneMats {
    alignas(16) float gb[16];
    __m128 gn_c0, gn_c1, gn_c2;
    int currentLight;
};

enum Uninitialized { UNINITIALIZED };

struct alignas(16) Mat4 {
    float m[16];

    Mat4() { identity(); }

    Mat4(Uninitialized) {
    }

    Mat4(const float *data) { std::memcpy(m, data, 16 * sizeof(float)); }

    inline void identity() {
        std::memset(m, 0, 16 * sizeof(float));
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    inline void mul(const Mat4 &right) {
        __m128 l0 = _mm_load_ps(&m[0]);
        __m128 l1 = _mm_load_ps(&m[4]);
        __m128 l2 = _mm_load_ps(&m[8]);
        __m128 l3 = _mm_load_ps(&m[12]);

        auto mac = [&](const float *r_col) {
            __m128 v = _mm_load_ps(r_col);
            __m128 res = _mm_mul_ps(l0, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)));
            res = MADD_PS(l1, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)), res);
            res = MADD_PS(l2, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)), res);
            res = MADD_PS(l3, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)), res);
            return res;
        };

        __m128 r0 = mac(&right.m[0]);
        __m128 r1 = mac(&right.m[4]);
        __m128 r2 = mac(&right.m[8]);
        __m128 r3 = mac(&right.m[12]);

        _mm_store_ps(&m[0], r0);
        _mm_store_ps(&m[4], r1);
        _mm_store_ps(&m[8], r2);
        _mm_store_ps(&m[12], r3);
    }

    inline Mat4 normalMatrix4x4() const {
        Mat4 res(UNINITIALIZED);
        res.m[0] = m[0];
        res.m[1] = m[1];
        res.m[2] = m[2];
        res.m[3] = 0.0f;
        res.m[4] = m[4];
        res.m[5] = m[5];
        res.m[6] = m[6];
        res.m[7] = 0.0f;
        res.m[8] = m[8];
        res.m[9] = m[9];
        res.m[10] = m[10];
        res.m[11] = 0.0f;
        res.m[12] = 0.0f;
        res.m[13] = 0.0f;
        res.m[14] = 0.0f;
        res.m[15] = 1.0f;
        return res;
    }
};

// Quad buckets: the shipped binary splits by (cullable, translucent) at parse
// time [C] and consumes the buckets back-to-back per model render.
struct QuadBuckets {
    std::vector<FastQuad> cullable;                    // opaque first
    std::vector<FastQuad> nonCullable;
    std::vector<FastQuad> cullableTranslucent;
    std::vector<FastQuad> nonCullableTranslucent;

    size_t size() const {
        return cullable.size() + nonCullable.size() +
               cullableTranslucent.size() + nonCullableTranslucent.size();
    }
};

struct NativeModel {
    std::vector<NativeBone> bones;
    QuadBuckets buckets;
    std::vector<int> evalOrder;
    std::vector<int> evalPos; // evalOrder index per bone (cycle defence)

    std::vector<Mat4> cacheGlobalTransforms;
    std::vector<Mat4> cacheGlobalNormals;
    std::vector<PrecomputedBoneMats> cachePrecompMats;
    std::vector<int> visibleBones;
};

struct GpuVertex {
    float pos[3];
    float uv[2];
    uint32_t normal;
    uint16_t boneId;
    uint8_t partMask;
    uint8_t flags;
    uint32_t pad;
};

static_assert(sizeof(GpuVertex) == 32, "GpuVertex size error");

struct BoneDataOut {
    float transform[16];
    float normal[16];
    int32_t packedLight;
    int32_t isHidden;
    int32_t pad[2];
};

static_assert(sizeof(BoneDataOut) == 144, "BoneDataOut mismatch");

struct NativeGpuMesh {
    std::vector<NativeBone> bones;
    std::vector<int> evalOrder;
    std::vector<int> evalPos; // evalOrder index per bone (cycle defence)
    int boneCount = 0;
    std::unique_ptr<GpuVertex[]> vertexData;
    std::unique_ptr<uint32_t[]> indexData;
    int vertexCount = 0;
    int indexCount = 0;
    int partMask1Start = 0, partMask1Count = 0;
    int partMask2Start = 0, partMask2Count = 0;
    int partMask3Start = 0, partMask3Count = 0;
    std::vector<Mat4> globalTransforms;
    std::vector<Mat4> globalNormals;
    std::vector<uint8_t> hiddenInherited;
};

static inline uint32_t packNormal_2_10_10_10_REV(float x, float y, float z) {
    auto pack = [](float v) -> uint32_t {
        int i = static_cast<int>(std::lround(v * 511.0f));
        if (i < -512) i = -512;
        if (i > 511) i = 511;
        return static_cast<uint32_t>(i & 0x3FF);
    };
    return pack(x) | (pack(y) << 10) | (pack(z) << 20);
}

// ---------------------------------------------------------------------------
// Begin JNI exports.
// ---------------------------------------------------------------------------
extern "C" {

// ---------------------------------------------------------------------------
// [A] nInitSIMD — rebuilt from GM_nInitSIMD.asm:1-332.
//
// Java contract (GeoModel.java:198-220):
//   nInitSIMD(Class bufferBuilderClass, String bufferName, String verticesName,
//             String nextElementByteName, String ensureCapacityName,
//             String modeName, Class vertexFormatModeClass)
//
// Asm trace:
//   :12-32   null-check all 7 args; on failure FindClass@0x30
//            ("java/lang/IllegalArgumentException", rodata 0xa12b) +
//            ThrowNew@0x70 ("[OpenYSM.cpp] Missing required arguments for
//            nInitSIMD", rodata 0xa7ff).
//   :62-73   GetStringUTFChars@0x548 for the five names.
//   :77-89   GetObjectClass@0xf8(Mode.class) -> java/lang/Class;
//            GetMethodID@0x108(Class, "getName"(0xa46a),
//            "()Ljava/lang/String;"(0xa9c3)); DeleteLocalRef@0xb8.
//   :97-115  helper 1dde0 = CallObjectMethodV trampoline (full.asm:8167) ->
//            Mode.class.getName(); GetStringUTFChars on the result.
//   :115-142 build descriptor: 'L' (0x4c @1bffa) + name with '.'->'/'
//            (0x2e->0x2f @1c01f-1c031) + ';' (0x3b @1c071).
//   :168-178 DeleteGlobalRef@0xb0 old, NewGlobalRef@0xa8 new BufferBuilder
//            class.
//   :180-236 GetFieldID@0x2f0: buffer -> "Ljava/nio/ByteBuffer;"(0xa95f),
//            vertices / nextElementByte -> "I"(0xa7e9), mode -> descriptor;
//            GetMethodID@0x108: ensureCapacity -> "(I)V"(0xa67c);
//            ExceptionCheck@0x720 between each.
//   :243-275 FindClass@0x30 SodiumBufferBuilder(0xa036) — optional: absent
//            class -> ExceptionClear@0x88 and globals stay null; present ->
//            NewGlobalRef@0xa8 + GetFieldID@0x2f0("builder"(0xa02e),
//            "Lme/jellysquid/.../ExtendedBufferBuilder;"(0xa975)).
//   :283-324 ReleaseStringUTFChars@0x550 x5.
// ---------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nInitSIMD(
    JNIEnv *env, jclass clazz, jclass bufferBuilderClass, jstring bufferName,
    jstring verticesName, jstring nextElementByteName, jstring ensureCapacityName,
    jstring modeName, jclass vertexFormatModeClass) {
    if (!bufferBuilderClass || !bufferName || !verticesName || !nextElementByteName ||
        !ensureCapacityName || !modeName || !vertexFormatModeClass) {
        jclass ex = env->FindClass("java/lang/IllegalArgumentException");
        if (ex) {
            env->ThrowNew(ex, "[OpenYSM.cpp] Missing required arguments for nInitSIMD");
        }
        return;
    }

    const char *bufferNameC = env->GetStringUTFChars(bufferName, nullptr);
    const char *verticesNameC = bufferNameC ? env->GetStringUTFChars(verticesName, nullptr) : nullptr;
    const char *nextElementByteNameC =
        verticesNameC ? env->GetStringUTFChars(nextElementByteName, nullptr) : nullptr;
    const char *ensureCapacityNameC =
        nextElementByteNameC ? env->GetStringUTFChars(ensureCapacityName, nullptr) : nullptr;
    const char *modeNameC =
        ensureCapacityNameC ? env->GetStringUTFChars(modeName, nullptr) : nullptr;

    // Build the field descriptor of the Mode enum class at runtime so
    // remapped game jars keep working (asm 1bf3d-1c076).
    std::string modeDescriptor;
    jclass classClass = env->GetObjectClass(vertexFormatModeClass);
    jmethodID getNameID = classClass
                              ? env->GetMethodID(classClass, "getName", "()Ljava/lang/String;")
                              : nullptr;
    if (classClass) env->DeleteLocalRef(classClass);
    if (getNameID && !env->ExceptionCheck()) {
        jstring modeClassName = (jstring) env->CallObjectMethod(vertexFormatModeClass, getNameID);
        if (modeClassName && !env->ExceptionCheck()) {
            const char *cn = env->GetStringUTFChars(modeClassName, nullptr);
            if (cn) {
                modeDescriptor += 'L';
                for (const char *p = cn; *p; ++p) {
                    modeDescriptor += (*p == '.') ? '/' : *p;
                }
                modeDescriptor += ';';
                env->ReleaseStringUTFChars(modeClassName, cn);
            }
            env->DeleteLocalRef(modeClassName);
        } else if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }

    if (!modeDescriptor.empty()) {
        if (g_BufferBuilderClass) env->DeleteGlobalRef(g_BufferBuilderClass);
        g_BufferBuilderClass = (jclass) env->NewGlobalRef(bufferBuilderClass);

        // [A2] Per-field runtime probes, each exception-cleared on miss. The
        // legacy surface arms only when nextElementByte + ensureCapacity both
        // resolve; the staging surface arms when the buffer field probe falls
        // through to the ByteBufferBuilder descriptor (26.2/26.3). mode is a
        // pure init gate (never read back by the fast paths) so its miss is
        // always tolerated — this is what makes 26.x init succeed at all.
        g_bufferFieldID =
            env->GetFieldID(g_BufferBuilderClass, bufferNameC, "Ljava/nio/ByteBuffer;");
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_bufferFieldID = env->GetFieldID(
                g_BufferBuilderClass, bufferNameC,
                "Lcom/mojang/blaze3d/vertex/ByteBufferBuilder;");
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                g_bufferFieldID = nullptr;
                g_stagingBufferField = false;
            } else {
                g_stagingBufferField = true;
            }
        } else {
            g_stagingBufferField = false;
        }
        g_verticesFieldID = env->GetFieldID(g_BufferBuilderClass, verticesNameC, "I");
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_verticesFieldID = nullptr;
        }
        g_nextElementByteFieldID =
            env->GetFieldID(g_BufferBuilderClass, nextElementByteNameC, "I");
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_nextElementByteFieldID = nullptr;
        }
        g_modeFieldID =
            env->GetFieldID(g_BufferBuilderClass, modeNameC, modeDescriptor.c_str());
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_modeFieldID = nullptr;
        }
        g_ensureCapacityMethodID =
            env->GetMethodID(g_BufferBuilderClass, ensureCapacityNameC, "(I)V");
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_ensureCapacityMethodID = nullptr;
        }
        // Staging write guard: the fast path emits fixed 36-byte NEW_ENTITY
        // records, so it must only engage when the live builder strides 36
        // bytes (DefaultVertexFormat.ENTITY). Optional — the legacy surface
        // has no such field and the probe miss is tolerated.
        g_vertexSizeFieldID = env->GetFieldID(g_BufferBuilderClass, "vertexSize", "I");
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_vertexSizeFieldID = nullptr;
        }

        // Reset lazy staging state on re-init.
        g_reserveMethodID = nullptr;
        g_reserveProbeDead = false;

        // Optional Sodium integration; the shipped binary clears the pending
        // exception when the class is missing (asm 1c1e3-1c254).
        jclass sodium = env->FindClass(
            "me/jellysquid/mods/sodium/client/render/vertex/buffer/SodiumBufferBuilder");
        if (sodium) {
            g_SodiumBufferBuilderClass = (jclass) env->NewGlobalRef(sodium);
            g_sodiumBuilderFieldID = env->GetFieldID(
                g_SodiumBufferBuilderClass, "builder",
                "Lme/jellysquid/mods/sodium/client/render/vertex/buffer/ExtendedBufferBuilder;");
            if (!g_sodiumBuilderFieldID) env->ExceptionClear();
            env->DeleteLocalRef(sodium);
        } else {
            env->ExceptionClear();
        }
    }

    if (bufferNameC) env->ReleaseStringUTFChars(bufferName, bufferNameC);
    if (ensureCapacityNameC) env->ReleaseStringUTFChars(ensureCapacityName, ensureCapacityNameC);
    if (nextElementByteNameC)
        env->ReleaseStringUTFChars(nextElementByteName, nextElementByteNameC);
    if (verticesNameC) env->ReleaseStringUTFChars(verticesName, verticesNameC);
    if (modeNameC) env->ReleaseStringUTFChars(modeName, modeNameC);
}

JNIEXPORT jlong JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nInitModelCache(
    JNIEnv *env, jclass clazz, jobject buffer) {
    char *data = (char *) env->GetDirectBufferAddress(buffer);
    if (!data) return 0;

    NativeModel *model = new NativeModel();
    int offset = 0;

    auto readInt = [&]() {
        int v;
        std::memcpy(&v, data + offset, 4);
        offset += 4;
        return v;
    };
    auto readFloat = [&]() {
        float v;
        std::memcpy(&v, data + offset, 4);
        offset += 4;
        return v;
    };
    auto readByte = [&]() {
        char v = data[offset];
        offset += 1;
        return v;
    };

    int boneCount = readInt();
    model->bones.resize(boneCount);
    model->cacheGlobalTransforms.resize(boneCount);
    model->cacheGlobalNormals.resize(boneCount);
    model->cachePrecompMats.resize(boneCount);
    model->visibleBones.reserve(boneCount);
    model->evalPos.assign(boneCount, -1);

    std::vector<std::vector<int> > children(boneCount);

    for (int i = 0; i < boneCount; ++i) {
        NativeBone &bone = model->bones[i];
        bone.parentIdx = readInt();
        // Non-tree defence: an out-of-range parent would OOB-write children[]
        // here and leave bones unreachable below.
        if (bone.parentIdx < -1 || bone.parentIdx >= boneCount) bone.parentIdx = -1;
        if (bone.parentIdx != -1) {
            children[bone.parentIdx].push_back(i);
        }

        bone.partMask = readInt();
        bone.glow = readByte() != 0;
        bone.pivotX = readFloat();
        bone.pivotY = readFloat();
        bone.pivotZ = readFloat();

        bone.quadStart = 0;

        __m128 vminX = _mm_set1_ps(INFINITY), vmaxX = _mm_set1_ps(-INFINITY);
        __m128 vminY = vminX, vmaxY = vmaxX;
        __m128 vminZ = vminX, vmaxZ = vmaxX;

        int cubeCount = readInt();
        for (int j = 0; j < cubeCount; ++j) {
            bool cullable = readByte() != 0;
            int quadCount = readInt();
            for (int k = 0; k < quadCount; ++k) {
                // [C] The shipped binary consumes this byte: it selects which
                // of the four quad buckets the quad lands in
                // (GM_nInitModelCache.asm:375 movzbl + :436-469).
                bool translucent = readByte() != 0;

                FastQuad fq;
                fq.boneIdx = i;
                fq.cullable = cullable;

                alignas(16) float tmpX[4], tmpY[4], tmpZ[4], tmpU[4], tmpV[4];
                for (int v = 0; v < 4; ++v) {
                    tmpX[v] = readFloat();
                    tmpY[v] = readFloat();
                    tmpZ[v] = readFloat();
                }
                for (int v = 0; v < 4; ++v) {
                    tmpU[v] = readFloat();
                    tmpV[v] = readFloat();
                }
                fq.nx = readFloat();
                fq.ny = readFloat();
                fq.nz = readFloat();

                fq.x = _mm_load_ps(tmpX);
                fq.y = _mm_load_ps(tmpY);
                fq.z = _mm_load_ps(tmpZ);
                fq.u = _mm_load_ps(tmpU);
                fq.v = _mm_load_ps(tmpV);

                vminX = _mm_min_ps(vminX, fq.x);
                vmaxX = _mm_max_ps(vmaxX, fq.x);
                vminY = _mm_min_ps(vminY, fq.y);
                vmaxY = _mm_max_ps(vmaxY, fq.y);
                vminZ = _mm_min_ps(vminZ, fq.z);
                vmaxZ = _mm_max_ps(vmaxZ, fq.z);

                if (cullable) {
                    if (!translucent) {
                        model->buckets.cullable.push_back(fq);
                    } else {
                        model->buckets.cullableTranslucent.push_back(fq);
                    }
                } else {
                    if (!translucent) {
                        model->buckets.nonCullable.push_back(fq);
                    } else {
                        model->buckets.nonCullableTranslucent.push_back(fq);
                    }
                }
            }
        }
        bone.quadCount = (int) model->buckets.size() - bone.quadStart;

        if (bone.quadCount > 0) {
            auto hmin = [](__m128 v) {
                v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
                v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtss_f32(v);
            };
            auto hmax = [](__m128 v) {
                v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
                v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtss_f32(v);
            };
            bone.aabbMin[0] = hmin(vminX);
            bone.aabbMax[0] = hmax(vmaxX);
            bone.aabbMin[1] = hmin(vminY);
            bone.aabbMax[1] = hmax(vmaxY);
            bone.aabbMin[2] = hmin(vminZ);
            bone.aabbMax[2] = hmax(vmaxZ);
        } else {
            bone.aabbMin[0] = bone.aabbMin[1] = bone.aabbMin[2] = 0.0f;
            bone.aabbMax[0] = bone.aabbMax[1] = bone.aabbMax[2] = 0.0f;
        }
    }

    // Eval order = pre-order DFS from every root. Non-tree defence: cycles
    // and orphaned bones would stay unvisited, and the render loop indexes
    // evalOrder[0..boneCount) -> OOB read (probed SIGSEGV). Guarded DFS
    // appends leftovers as pseudo-roots; identical output for valid trees.
    std::vector<char> visited(boneCount, 0);
    std::function<int(int)> dfs = [&](int idx) -> int {
        if (visited[idx]) return 0;
        visited[idx] = 1;
        model->evalPos[idx] = (int) model->evalOrder.size();
        model->evalOrder.push_back(idx);
        int count = 0;
        for (int child: children[idx]) {
            count += dfs(child);
        }
        model->bones[idx].subtreeCount = count;
        return count + 1;
    };

    for (int i = 0; i < boneCount; ++i) {
        if (model->bones[i].parentIdx == -1) dfs(i);
    }
    for (int i = 0; i < boneCount; ++i) {
        if (!visited[i]) dfs(i);
    }

    return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nDestroyModelCache(
    JNIEnv *env, jclass clazz, jlong handle) {
    delete reinterpret_cast<NativeModel *>(handle);
}

// Per-vertex 36-byte NEW_ENTITY record, byte-decoded from the shipped
// binary's fast-path output (harness fast.buffer dumps):
//   pos 3xf32 | color 4xu8 = (u8)(c*255) trunc | uv 2xf32
//   overlay 2xs16 + light 2xs16 (raw ovl_light64 split)
//   normal 3xs8 = (s8)(n*127) trunc | pad 0x00
// Shared by the legacy fast path (write target = buffer + nextElementByte) and
// the 26.x staging fast path (write target = ByteBufferBuilder.reserve ptr).
static size_t writeNewEntityQuads(uint8_t *out, const std::vector<float> &fData,
                                  const std::vector<uint64_t> &quadOvlLight, int actualQuads) {
    size_t cursor = 0;
    for (int q = 0; q < actualQuads; ++q) {
        const float *quad = fData.data() + (size_t) q * 48;
        const uint64_t ovl = quadOvlLight[q];
        for (int v = 0; v < 4; ++v) {
            const float *src = quad + v * 12;
            uint8_t *o = out + cursor;
            std::memcpy(o, src, 3 * sizeof(float));
            o[12] = (uint8_t) (int) (src[3] * 255.0f);
            o[13] = (uint8_t) (int) (src[4] * 255.0f);
            o[14] = (uint8_t) (int) (src[5] * 255.0f);
            o[15] = (uint8_t) (int) (src[6] * 255.0f);
            std::memcpy(o + 16, src + 7, 2 * sizeof(float));
            std::memcpy(o + 24, &ovl, sizeof(uint64_t));
            o[32] = (uint8_t) (int8_t) (src[9] * 127.0f);
            o[33] = (uint8_t) (int8_t) (src[10] * 127.0f);
            o[34] = (uint8_t) (int8_t) (src[11] * 127.0f);
            o[35] = 0;
            cursor += 36;
        }
    }
    return cursor;
}

// [B] Registered descriptor "(JLjava/lang/Object;[F[F[FIIIFFFF)V" — the extra
// [F (stateArray) sits right after animArray, matching GeoModel.java:226-236.
JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nComputeModelVertices(
    JNIEnv *env, jclass clazz, jlong handle, jobject vertexConsumer,
    jfloatArray matrixArray, jfloatArray animArray, jfloatArray stateArray,
    jint renderPartMask, jint packedLight, jint packedOverlay,
    jfloat r, jfloat g, jfloat b, jfloat a) {
    NativeModel *model = reinterpret_cast<NativeModel *>(handle);
    if (!model || model->buckets.size() == 0) return;

    // [A] Direct-write fast-path availability (asm 16543-165eb):
    // vertexConsumer != null && a surface armed && optional Sodium unwrap &&
    // IsInstanceOf(cached BufferBuilder class). Legacy surface = ensureCapacity
    // + nextElementByte cached; staging surface = buffer field probed as
    // ByteBufferBuilder (26.2/26.3), reserve resolved lazily at first write.
    bool builderAvailable = false;
    bool stagingSurface = false;
    jobject builder = vertexConsumer;
    if (vertexConsumer && g_BufferBuilderClass && g_bufferFieldID && g_verticesFieldID &&
        ((g_nextElementByteFieldID && g_ensureCapacityMethodID) || g_stagingBufferField)) {
        if (g_SodiumBufferBuilderClass && g_sodiumBuilderFieldID &&
            env->IsInstanceOf(vertexConsumer, g_SodiumBufferBuilderClass)) {
            jobject inner = env->GetObjectField(vertexConsumer, g_sodiumBuilderFieldID);
            if (inner) builder = inner;
        }
        builderAvailable = env->IsInstanceOf(builder, g_BufferBuilderClass);
        stagingSurface = builderAvailable && g_stagingBufferField && !g_nextElementByteFieldID;
        // 36B-record guard: a non-entity stride would corrupt the mesh —
        // fall back to the stride-agnostic submitVertices callback path.
        if (stagingSurface && g_vertexSizeFieldID &&
            env->GetIntField(builder, g_vertexSizeFieldID) != 36) {
            stagingSurface = false;
            builderAvailable = false;
        }
    }

    jfloat *matricesData;
    jfloat *animData;
    if (builderAvailable) {
        matricesData = env->GetFloatArrayElements(matrixArray, nullptr);
        animData = env->GetFloatArrayElements(animArray, nullptr);
    } else {
        matricesData = (jfloat *) env->GetPrimitiveArrayCritical(matrixArray, nullptr);
        animData = (jfloat *) env->GetPrimitiveArrayCritical(animArray, nullptr);
    }

    Mat4 rootPoseMat(matricesData);
    float *rootNormalArr = matricesData + 16;
    Mat4 projMat(matricesData + 32);

    const bool cullingEnabled = std::fabs(projMat.m[11]) > 1e-3f;

    alignas(16) float frustum[6][4];
    if (cullingEnabled) {
        const float *M = projMat.m;
        for (int j = 0; j < 4; ++j) {
            float r0 = M[j * 4 + 0];
            float r1 = M[j * 4 + 1];
            float r2 = M[j * 4 + 2];
            float r3 = M[j * 4 + 3];
            frustum[0][j] = r3 + r0;
            frustum[1][j] = r3 - r0;
            frustum[2][j] = r3 + r1;
            frustum[3][j] = r3 - r1;
            frustum[4][j] = r3 + r2;
            frustum[5][j] = r3 - r2;
        }
    }

    Mat4 rootNormalMat(UNINITIALIZED);
    rootNormalMat.m[0] = rootNormalArr[0];
    rootNormalMat.m[1] = rootNormalArr[1];
    rootNormalMat.m[2] = rootNormalArr[2];
    rootNormalMat.m[3] = 0.0f;
    rootNormalMat.m[4] = rootNormalArr[3];
    rootNormalMat.m[5] = rootNormalArr[4];
    rootNormalMat.m[6] = rootNormalArr[5];
    rootNormalMat.m[7] = 0.0f;
    rootNormalMat.m[8] = rootNormalArr[6];
    rootNormalMat.m[9] = rootNormalArr[7];
    rootNormalMat.m[10] = rootNormalArr[8];
    rootNormalMat.m[11] = 0.0f;
    rootNormalMat.m[12] = 0.0f;
    rootNormalMat.m[13] = 0.0f;
    rootNormalMat.m[14] = 0.0f;
    rootNormalMat.m[15] = 1.0f;

    const int glowLight = (15 << 4) | (15 << 20);
    const size_t boneCount = model->bones.size();

    model->visibleBones.clear();

    // [B] Sentinel scan (asm 16649-16673): stateArray mode activates only when
    // some bone's animData[bone*12+11] equals 1.0f (rodata 9c50).
    bool useStateArray = false;
    for (size_t i = 0; i < boneCount; ++i) {
        if (animData[i * 12 + 11] == 1.0f) {
            useStateArray = true;
            break;
        }
    }

    jsize stateLen = 0;
    jfloat *stateData = nullptr;
    if (useStateArray && stateArray) {
        stateLen = env->GetArrayLength(stateArray);
        if (builderAvailable) {
            stateData = env->GetFloatArrayElements(stateArray, nullptr);
        } else {
            stateData = (jfloat *) env->GetPrimitiveArrayCritical(stateArray, nullptr);
        }
    }

    // Quad staging: visible bones, partMask filter, bucket order preserved.
    struct Staged {
        const FastQuad *fq;
        const PrecomputedBoneMats *pMat;
        bool cullable;
        bool translucent;
    };
    static thread_local std::vector<Staged> staged;
    staged.clear();

    const __m128 rgba = _mm_setr_ps(r, g, b, a);

    int k = 0;
    while (k < (int) boneCount) {
        int bIdx = model->evalOrder[k];
        NativeBone &bone = model->bones[bIdx];

        int pOffset = bIdx * 12;
        float animRx = animData[pOffset + 0], animRy = animData[pOffset + 1], animRz = animData[pOffset + 2];
        float animTx = animData[pOffset + 3], animTy = animData[pOffset + 4], animTz = animData[pOffset + 5];
        float animSx = animData[pOffset + 6], animSy = animData[pOffset + 7], animSz = animData[pOffset + 8];
        // Skip flag from animData[+10]. The shipped binary ALSO skips the
        // bone itself and its subtree when any scale slot is 0 (probed:
        // mid-chain scale0 -> self+child quads gone with culling disabled).
        float skipChildrenFlag = animData[pOffset + 10];
        const bool zeroScale = (animSx == 0.0f || animSy == 0.0f || animSz == 0.0f);
        // native-dll-round1: offset9 (HIDDEN) consumption — the shipped
        // binary ignores it (probed), the CPU path (calculateBoneMatrix)
        // folds it into visibleCache which propagates to the subtree.
        // Mirrors that: hidden bone emits no quads, subtree skipped. This
        // replaces the temporary Java-side hiddenPatchedBoneParams fallback.
        const bool hiddenSelf = animData[pOffset + 9] != 0.0f;

        float px = bone.pivotX * 0.0625f, py = bone.pivotY * 0.0625f, pz = bone.pivotZ * 0.0625f;
        float dx = px - animTx * 0.0625f;
        float dy = py + animTy * 0.0625f;
        float dz = pz + animTz * 0.0625f;

        float cx, sx, cy, sy, cz, sz;
        FAST_SINCOS(animRx, &sx, &cx);
        FAST_SINCOS(animRy, &sy, &cy);
        FAST_SINCOS(animRz, &sz, &cz);

        Mat4 localMat(UNINITIALIZED);
        localMat.m[0] = (cz * cy) * animSx;
        localMat.m[1] = (sz * cy) * animSx;
        localMat.m[2] = (-sy) * animSx;
        localMat.m[3] = 0.0f;

        localMat.m[4] = (cz * sy * sx - sz * cx) * animSy;
        localMat.m[5] = (sz * sy * sx + cz * cx) * animSy;
        localMat.m[6] = (cy * sx) * animSy;
        localMat.m[7] = 0.0f;

        localMat.m[8] = (cz * sy * cx + sz * sx) * animSz;
        localMat.m[9] = (sz * sy * cx - cz * sx) * animSz;
        localMat.m[10] = (cy * cx) * animSz;
        localMat.m[11] = 0.0f;

        localMat.m[12] = dx - (localMat.m[0] * px + localMat.m[4] * py + localMat.m[8] * pz);
        localMat.m[13] = dy - (localMat.m[1] * px + localMat.m[5] * py + localMat.m[9] * pz);
        localMat.m[14] = dz - (localMat.m[2] * px + localMat.m[6] * py + localMat.m[10] * pz);
        localMat.m[15] = 1.0f;

        // Cycle defence: a parent that is not evaluated BEFORE this bone in
        // this pass (cycle/orphan topology) is treated as a root. Without
        // this, the cached transform from the PREVIOUS pass leaks in and the
        // output becomes pass-dependent (observed: det flips between passes).
        const bool parentReady =
            bone.parentIdx != -1 && model->evalPos[bone.parentIdx] < model->evalPos[bIdx];
        const Mat4 &parentGlobal =
            parentReady ? model->cacheGlobalTransforms[bone.parentIdx] : rootPoseMat;
        Mat4 &globalMat = model->cacheGlobalTransforms[bIdx];
        globalMat = parentGlobal;
        globalMat.mul(localMat);

        const Mat4 &parentNormal =
            parentReady ? model->cacheGlobalNormals[bone.parentIdx] : rootNormalMat;
        Mat4 localNormalMat = localMat.normalMatrix4x4();
        Mat4 &globalNormalMat = model->cacheGlobalNormals[bIdx];
        globalNormalMat = parentNormal;
        globalNormalMat.mul(localNormalMat);

        auto &precomp = model->cachePrecompMats[bIdx];
        std::memcpy(precomp.gb, globalMat.m, 16 * sizeof(float));

        precomp.gn_c0 = _mm_load_ps(&globalNormalMat.m[0]);
        precomp.gn_c1 = _mm_load_ps(&globalNormalMat.m[4]);
        precomp.gn_c2 = _mm_load_ps(&globalNormalMat.m[8]);
        precomp.currentLight = bone.glow ? glowLight : packedLight;

        // [B] stateArray write (asm 170d5-171c2): guard animData[+11] == 1.0f
        // (rodata 9c50) and bounds bIdx*4+2 < stateLen; transform uses the
        // parent's already-updated global (identity for roots, asm 170a4-17113)
        // and constants 9c40 = {-16, 16} / 9c58 = {16}.
        if (stateData && animData[pOffset + 11] == 1.0f && bIdx * 4 + 2 < stateLen) {
            static const Mat4 identity;
            const Mat4 &m = parentReady ? model->cacheGlobalTransforms[bone.parentIdx] : identity;
            float wx = m.m[0] * dx + m.m[4] * dy + m.m[8] * dz + m.m[12];
            float wy = m.m[1] * dx + m.m[5] * dy + m.m[9] * dz + m.m[13];
            float wz = m.m[2] * dx + m.m[6] * dy + m.m[10] * dz + m.m[14];
            stateData[bIdx * 4 + 0] = wx * -16.0f;
            stateData[bIdx * 4 + 1] = wy * 16.0f;
            stateData[bIdx * 4 + 2] = wz * 16.0f;
        }

        if (cullingEnabled && bone.quadCount > 0) {
            const float *M = globalMat.m;
            float bcx = (bone.aabbMax[0] + bone.aabbMin[0]) * 0.5f;
            float bcy = (bone.aabbMax[1] + bone.aabbMin[1]) * 0.5f;
            float bcz = (bone.aabbMax[2] + bone.aabbMin[2]) * 0.5f;
            float bex = (bone.aabbMax[0] - bone.aabbMin[0]) * 0.5f;
            float bey = (bone.aabbMax[1] - bone.aabbMin[1]) * 0.5f;
            float bez = (bone.aabbMax[2] - bone.aabbMin[2]) * 0.5f;
            float wMin[3], wMax[3];
            for (int row = 0; row < 3; ++row) {
                float m0 = M[0 * 4 + row], m1 = M[1 * 4 + row], m2 = M[2 * 4 + row], m3 = M[3 * 4 + row];
                float wc = m0 * bcx + m1 * bcy + m2 * bcz + m3;
                float we = std::fabs(m0) * bex + std::fabs(m1) * bey + std::fabs(m2) * bez;
                wMin[row] = wc - we;
                wMax[row] = wc + we;
            }
            bool outside = false;
            for (int p = 0; p < 6; ++p) {
                float fa = frustum[p][0], fb = frustum[p][1], fc = frustum[p][2], fd = frustum[p][3];
                float ppx = (fa >= 0.0f) ? wMax[0] : wMin[0];
                float ppy = (fb >= 0.0f) ? wMax[1] : wMin[1];
                float ppz = (fc >= 0.0f) ? wMax[2] : wMin[2];
                if (fa * ppx + fb * ppy + fc * ppz + fd < 0.0f) {
                    outside = true;
                    break;
                }
            }
            if (outside) {
                if (skipChildrenFlag != 0.0f || zeroScale || hiddenSelf) {
                    k += bone.subtreeCount + 1;
                } else {
                    k++;
                }
                continue;
            }
        }

        // Shipped binary: a zero-scale bone emits no quads and skips its
        // subtree (mirrors the CPU path's visibleCache propagation).
        // native-dll-round1: offset9 joins the same skip treatment.
        if (!zeroScale && !hiddenSelf) {
            model->visibleBones.push_back(bIdx);
        }

        if (skipChildrenFlag != 0.0f || zeroScale || hiddenSelf) {
            k += bone.subtreeCount + 1;
        } else {
            k++;
        }
    }

    // Stage quads of visible bones. The shipped binary consumes GLOBAL
    // buckets across all visible bones in this order (decoded from 4-bucket
    // probe + multi-bone case): nonCullable, cullable, nonCullableTranslucent,
    // cullableTranslucent; wire order within a bucket, bone order across bones.
    {
        auto collectBucket = [&](const std::vector<FastQuad> &bucket, bool cullable, bool translucent) {
            for (int bIdx: model->visibleBones) {
                const NativeBone &bone = model->bones[bIdx];
                if (bone.quadCount == 0) continue;
                if (renderPartMask != 0 && bone.partMask != renderPartMask && bone.partMask != 3)
                    continue;
                for (const FastQuad &fq: bucket)
                    if (fq.boneIdx == bIdx)
                        staged.push_back({&fq, &model->cachePrecompMats[bIdx], cullable, translucent});
            }
        };
        collectBucket(model->buckets.nonCullable, false, false);
        collectBucket(model->buckets.cullable, true, false);
        collectBucket(model->buckets.nonCullableTranslucent, false, true);
        collectBucket(model->buckets.cullableTranslucent, true, true);
    }

    int actualQuads = 0;

    static thread_local std::vector<float> fData;
    static thread_local std::vector<int> iData;
    static thread_local std::vector<uint64_t> quadOvlLight; // per output quad
    fData.clear();
    iData.clear();
    quadOvlLight.clear();
    fData.reserve(staged.size() * 48 + 16);
    iData.reserve(staged.size() * 8);
    quadOvlLight.reserve(staged.size());

    __m128 proj00 = _mm_set1_ps(projMat.m[0]), proj01 = _mm_set1_ps(projMat.m[4]),
           proj02 = _mm_set1_ps(projMat.m[8]), proj03 = _mm_set1_ps(projMat.m[12]);
    __m128 proj10 = _mm_set1_ps(projMat.m[1]), proj11 = _mm_set1_ps(projMat.m[5]),
           proj12 = _mm_set1_ps(projMat.m[9]), proj13 = _mm_set1_ps(projMat.m[13]);
    __m128 proj30 = _mm_set1_ps(projMat.m[3]), proj31 = _mm_set1_ps(projMat.m[7]),
           proj32 = _mm_set1_ps(projMat.m[11]), proj33 = _mm_set1_ps(projMat.m[15]);

    for (const Staged &st: staged) {
        const FastQuad &fq = *st.fq;
        const PrecomputedBoneMats &pMat = *st.pMat;

        const uint64_t ovl_light64 =
            (static_cast<uint64_t>(static_cast<uint32_t>(pMat.currentLight)) << 32) |
            static_cast<uint32_t>(packedOverlay);

        __m128 gb0 = _mm_set1_ps(pMat.gb[0]), gb1 = _mm_set1_ps(pMat.gb[1]), gb2 = _mm_set1_ps(pMat.gb[2]);
        __m128 gb4 = _mm_set1_ps(pMat.gb[4]), gb5 = _mm_set1_ps(pMat.gb[5]), gb6 = _mm_set1_ps(pMat.gb[6]);
        __m128 gb8 = _mm_set1_ps(pMat.gb[8]), gb9 = _mm_set1_ps(pMat.gb[9]), gb10 = _mm_set1_ps(pMat.gb[10]);
        __m128 gb12 = _mm_set1_ps(pMat.gb[12]), gb13 = _mm_set1_ps(pMat.gb[13]), gb14 = _mm_set1_ps(pMat.gb[14]);

        __m128 gX = MADD_PS(gb0, fq.x, MADD_PS(gb4, fq.y, MADD_PS(gb8, fq.z, gb12)));
        __m128 gY = MADD_PS(gb1, fq.x, MADD_PS(gb5, fq.y, MADD_PS(gb9, fq.z, gb13)));
        __m128 gZ = MADD_PS(gb2, fq.x, MADD_PS(gb6, fq.y, MADD_PS(gb10, fq.z, gb14)));

        if (st.cullable) {
            __m128 pX = MADD_PS(proj00, gX, MADD_PS(proj01, gY, MADD_PS(proj02, gZ, proj03)));
            __m128 pY = MADD_PS(proj10, gX, MADD_PS(proj11, gY, MADD_PS(proj12, gZ, proj13)));
            __m128 pW = MADD_PS(proj30, gX, MADD_PS(proj31, gY, MADD_PS(proj32, gZ, proj33)));

            __m128 pY_120 = _mm_shuffle_ps(pY, pY, _MM_SHUFFLE(3, 0, 2, 1));
            __m128 pW_201 = _mm_shuffle_ps(pW, pW, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 pY_201 = _mm_shuffle_ps(pY, pY, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 pW_120 = _mm_shuffle_ps(pW, pW, _MM_SHUFFLE(3, 0, 2, 1));

            __m128 sub = _mm_sub_ps(_mm_mul_ps(pY_120, pW_201), _mm_mul_ps(pY_201, pW_120));
            __m128 mx = _mm_mul_ps(pX, sub);
            __m128 mx1 = _mm_shuffle_ps(mx, mx, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 mx2 = _mm_shuffle_ps(mx, mx, _MM_SHUFFLE(2, 2, 2, 2));
            __m128 sum = _mm_add_ps(mx, _mm_add_ps(mx1, mx2));

            float det = _mm_cvtss_f32(sum);
            if (det <= 0.0f) continue;
        }

        __m128 n_res = MADD_PS(pMat.gn_c0, _mm_set1_ps(fq.nx),
                               MADD_PS(pMat.gn_c1, _mm_set1_ps(fq.ny),
                                       _mm_mul_ps(pMat.gn_c2, _mm_set1_ps(fq.nz))));
        __m128 dp = _mm_mul_ps(n_res, n_res);
        __m128 dsum = _mm_add_ps(dp, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(2, 3, 0, 1)));
        dsum = _mm_add_ps(dsum, _mm_shuffle_ps(dsum, dsum, _MM_SHUFFLE(1, 0, 3, 2)));
        n_res = _mm_mul_ps(n_res, _mm_rsqrt_ps(_mm_max_ps(dsum, _mm_set1_ps(1e-8f))));

        alignas(16) float fx[4], fy[4], fz[4], fu[4], fv[4], fn[4], frgba[4];
        _mm_store_ps(fx, gX);
        _mm_store_ps(fy, gY);
        _mm_store_ps(fz, gZ);
        _mm_store_ps(fu, fq.u);
        _mm_store_ps(fv, fq.v);
        _mm_store_ps(fn, n_res);
        _mm_store_ps(frgba, rgba);

        size_t base = fData.size();
        fData.resize(base + 48);
        for (int v = 0; v < 4; ++v) {
            float *fp = fData.data() + base + v * 12;
            fp[0] = fx[v];
            fp[1] = fy[v];
            fp[2] = fz[v];
            std::memcpy(fp + 3, frgba, 16);
            fp[7] = fu[v];
            fp[8] = fv[v];
            fp[9] = fn[0];
            fp[10] = fn[1];
            fp[11] = fn[2];

            // Resize FIRST, then write: memcpy into data()+size() followed by
            // resize() let the zero-init overwrite the payload (slow-path
            // overlay/light arrived all-zero — drift vs shipped caught by the
            // reworked harness).
            // Shipped-binary quirk, replicated for byte parity: on translucent
            // quads the FIRST vertex's overlay slot is 0 (light intact);
            // observed only in the slow path, the fast path writes both.
            const uint64_t ovl_light =
                (st.translucent && v == 0) ? (ovl_light64 & 0xFFFFFFFF00000000ULL) : ovl_light64;
            const size_t iBase = iData.size();
            iData.resize(iBase + 2);
            std::memcpy(iData.data() + iBase, &ovl_light, sizeof(uint64_t));
        }

        quadOvlLight.push_back(ovl_light64);
        actualQuads++;
    }

    int actualVertices = actualQuads * 4;

    // Release inputs. stateArray releases with mode 0 (copy back) — the
    // native side wrote it (asm 19ea0-19ebd); matrices/anim use JNI_ABORT.
    if (stateData) {
        if (builderAvailable) {
            env->ReleaseFloatArrayElements(stateArray, stateData, 0);
        } else {
            env->ReleasePrimitiveArrayCritical(stateArray, stateData, 0);
        }
    }
    if (builderAvailable) {
        env->ReleaseFloatArrayElements(matrixArray, matricesData, JNI_ABORT);
        env->ReleaseFloatArrayElements(animArray, animData, JNI_ABORT);
    } else {
        env->ReleasePrimitiveArrayCritical(matrixArray, matricesData, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(animArray, animData, JNI_ABORT);
    }

    if (actualVertices <= 0) return;

    bool fastWritten = false;
    if (builderAvailable && builder && !stagingSurface) {
        // [A] Legacy direct-write fast path (asm 189c9-18a54, 19e53-19e9a):
        //   ensureCapacity(staged quads * 144 bytes)       (pre-det-cull count:
        //                                                   shipped-binary probe
        //                                                   6 staged -> 864
        //                                                   while 5 emitted)
        //   vertices     = GetIntField(builder)            (189fa, 0x320)
        //   nextElementByte = GetIntField(builder)         (18a1e, 0x320)
        //   buffer       = GetObjectField + GetDirectBufferAddress
        //                                                  (18a3b-18a51)
        //   write quads*144 bytes at [addr + nextElementByte] (4 x 36B records)
        //   SetIntField(vertices, old + quads*4)           (19e56-19e76, 0x368)
        //   SetIntField(nextElementByte, old + quads*144)  (19e7c-19e9a, 0x368)
        env->CallVoidMethod(builder, g_ensureCapacityMethodID, (jint) (staged.size() * 144));
        if (env->ExceptionCheck()) return;

        jint vertices = env->GetIntField(builder, g_verticesFieldID);
        jint nextElementByte = env->GetIntField(builder, g_nextElementByteFieldID);
        jobject bufferObj = env->GetObjectField(builder, g_bufferFieldID);
        if (!bufferObj) return;
        void *addr = env->GetDirectBufferAddress(bufferObj);
        if (!addr) {
            env->DeleteLocalRef(bufferObj);
            return;
        }

        size_t cursor = writeNewEntityQuads((uint8_t *) addr + nextElementByte, fData,
                                            quadOvlLight, actualQuads);

        env->SetIntField(builder, g_verticesFieldID, (jint) (vertices + actualQuads * 4));
        env->SetIntField(builder, g_nextElementByteFieldID,
                         (jint) (nextElementByte + (jint) cursor));
        env->DeleteLocalRef(bufferObj);
        fastWritten = true;
    } else if (builderAvailable && builder && stagingSurface) {
        // [A2] 26.2/26.3 staging fast path. ByteBufferBuilder.reserve(int)
        // advances the shared staging writeOffset, grows the allocation and
        // returns the absolute write address (ByteBufferBuilder.java:49-55).
        // Reserve the EXACT post-cull payload (quads*144) — unlike
        // ensureCapacity, reserve consumes the bytes, so a pre-cull count
        // would strand unwritten holes in the mesh. The builder's vertex
        // count is then bumped; build()/storeMesh() derive the draw range
        // from writeOffset (slice) + vertices (DrawState index count).
        if (!g_reserveMethodID && !g_reserveProbeDead) {
            jobject bbuf = env->GetObjectField(builder, g_bufferFieldID);
            if (!bbuf) return;
            jclass cls = env->GetObjectClass(bbuf);
            if (cls) {
                g_reserveMethodID = env->GetMethodID(cls, "reserve", "(I)J");
                if (env->ExceptionCheck() || !g_reserveMethodID) {
                    env->ExceptionClear();
                    g_reserveMethodID = nullptr;
                    g_reserveProbeDead = true;
                }
                env->DeleteLocalRef(cls);
            }
            env->DeleteLocalRef(bbuf);
        }
        if (g_reserveMethodID) {
            jobject bbuf = env->GetObjectField(builder, g_bufferFieldID);
            if (!bbuf) return;
            jlong writePtr = env->CallLongMethod(bbuf, g_reserveMethodID,
                                                 (jint) (actualQuads * 144));
            if (env->ExceptionCheck()) {
                // Capacity overflow / closed buffer: nothing was written
                // (reserve does not advance on failure) — degrade to the
                // stride-agnostic callback path below.
                env->ExceptionClear();
            } else if (writePtr) {
                jint vertices = env->GetIntField(builder, g_verticesFieldID);
                writeNewEntityQuads((uint8_t *) writePtr, fData, quadOvlLight, actualQuads);
                env->SetIntField(builder, g_verticesFieldID, (jint) (vertices + actualQuads * 4));
                fastWritten = true;
            }
            env->DeleteLocalRef(bbuf);
        }
        // reserve lookup dead -> fastWritten stays false -> submitVertices.
    }
    if (!fastWritten && g_NativeModelRendererClass && g_submitVerticesID) {
        // Slow path: submitVertices callback (asm 19b0d-19e40).
        jobject fBuf = env->NewDirectByteBuffer(
            fData.data(), static_cast<jlong>(actualVertices) * 12 * sizeof(float));
        jobject iBuf = env->NewDirectByteBuffer(
            iData.data(), static_cast<jlong>(actualVertices) * 2 * sizeof(int));
        env->CallStaticVoidMethod(g_NativeModelRendererClass, g_submitVerticesID, vertexConsumer,
                                  actualVertices, fBuf, iBuf);
        env->DeleteLocalRef(fBuf);
        env->DeleteLocalRef(iBuf);
    }
}

JNIEXPORT jlong JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nBuildGpuMesh(
    JNIEnv *env, jclass clazz, jobject buffer, jintArray outMeta) {
    const uint8_t *data = reinterpret_cast<const uint8_t *>(env->GetDirectBufferAddress(buffer));
    if (!data) return 0;

    int offset = 0;
    auto readInt = [&]() {
        int v;
        std::memcpy(&v, data + offset, 4);
        offset += 4;
        return v;
    };
    auto readFloat = [&]() {
        float v;
        std::memcpy(&v, data + offset, 4);
        offset += 4;
        return v;
    };
    auto readByte = [&]() {
        uint8_t v = data[offset];
        offset += 1;
        return v;
    };

    auto *mesh = new NativeGpuMesh();
    int boneCount = readInt();
    mesh->boneCount = boneCount;
    mesh->bones.resize(boneCount);
    mesh->globalTransforms.resize(boneCount);
    mesh->globalNormals.resize(boneCount);
    mesh->hiddenInherited.assign(boneCount, 0);
    mesh->evalPos.assign(boneCount, -1);

    std::vector<std::vector<int> > children(boneCount);

    struct QuadRecord {
        uint32_t vertexOffset;
        uint8_t partMask;
    };
    std::vector<QuadRecord> quadRecords;
    quadRecords.reserve(1024);

    std::vector<GpuVertex> tmpVerts;
    tmpVerts.reserve(4096);

    for (int i = 0; i < boneCount; ++i) {
        NativeBone &bone = mesh->bones[i];
        bone.parentIdx = readInt();
        // Non-tree defence, same as nInitModelCache.
        if (bone.parentIdx < -1 || bone.parentIdx >= boneCount) bone.parentIdx = -1;
        if (bone.parentIdx != -1) children[bone.parentIdx].push_back(i);
        bone.partMask = readInt();
        bone.glow = readByte() != 0;
        bone.pivotX = readFloat();
        bone.pivotY = readFloat();
        bone.pivotZ = readFloat();
        bone.quadStart = 0;
        bone.quadCount = 0;
        bone.aabbMin[0] = bone.aabbMin[1] = bone.aabbMin[2] = 0.0f;
        bone.aabbMax[0] = bone.aabbMax[1] = bone.aabbMax[2] = 0.0f;

        int cubeCount = readInt();
        for (int c = 0; c < cubeCount; ++c) {
            uint8_t cullable = readByte() != 0 ? 1 : 0;
            int qc = readInt();
            for (int q = 0; q < qc; ++q) {
                uint32_t vOff = static_cast<uint32_t>(tmpVerts.size());

                // GPU wire has NO translucent byte: the quad stride is 92
                // bytes (12 pos + 8 uv + 3 normal floats). Evidence: shipped
                // binary GM_nBuildGpuMesh.asm quad advance = 0x5c; feeding a
                // 93B wire desyncs and crashes it. Writer counterpart =
                // GpuMeshBuilder.serializeModel (4+25B+5C+92Q capacity).

                float vx[4], vy[4], vz[4], uu[4], vv[4];
                for (int v = 0; v < 4; ++v) {
                    vx[v] = readFloat();
                    vy[v] = readFloat();
                    vz[v] = readFloat();
                }
                for (int v = 0; v < 4; ++v) {
                    uu[v] = readFloat();
                    vv[v] = readFloat();
                }
                float nx = readFloat(), ny = readFloat(), nz = readFloat();
                uint32_t packedNorm = packNormal_2_10_10_10_REV(nx, ny, nz);

                for (int v = 0; v < 4; ++v) {
                    GpuVertex gv;
                    gv.pos[0] = vx[v];
                    gv.pos[1] = vy[v];
                    gv.pos[2] = vz[v];
                    gv.uv[0] = uu[v];
                    gv.uv[1] = vv[v];
                    gv.normal = packedNorm;
                    gv.boneId = static_cast<uint16_t>(i);
                    gv.partMask = static_cast<uint8_t>(bone.partMask);
                    gv.flags = cullable;
                    gv.pad = 0;
                    tmpVerts.push_back(gv);
                }
                quadRecords.push_back({vOff, static_cast<uint8_t>(bone.partMask)});
            }
        }
    }

    // Non-tree defence (see nInitModelCache): guarded DFS, leftovers appended.
    std::vector<char> visited(boneCount, 0);
    std::function<int(int)> dfs = [&](int idx) -> int {
        if (visited[idx]) return 0;
        visited[idx] = 1;
        mesh->evalPos[idx] = (int) mesh->evalOrder.size();
        mesh->evalOrder.push_back(idx);
        int count = 0;
        for (int child: children[idx]) count += dfs(child);
        mesh->bones[idx].subtreeCount = count;
        return count + 1;
    };
    for (int i = 0; i < boneCount; ++i) {
        if (mesh->bones[i].parentIdx == -1) dfs(i);
    }
    for (int i = 0; i < boneCount; ++i) {
        if (!visited[i]) dfs(i);
    }

    std::stable_sort(quadRecords.begin(), quadRecords.end(),
                     [](const QuadRecord &a, const QuadRecord &b) { return a.partMask < b.partMask; });

    mesh->vertexCount = static_cast<int>(tmpVerts.size());
    mesh->vertexData.reset(new GpuVertex[mesh->vertexCount]);
    std::memcpy(mesh->vertexData.get(), tmpVerts.data(),
                static_cast<size_t>(mesh->vertexCount) * sizeof(GpuVertex));

    mesh->indexCount = static_cast<int>(quadRecords.size()) * 6;
    mesh->indexData.reset(new uint32_t[mesh->indexCount]);

    int currentPartMask = -1;
    int rangeStart = 0;
    auto closeRange = [&](int endIdx) {
        if (currentPartMask < 0) return;
        int count = endIdx - rangeStart;
        switch (currentPartMask) {
            case 1: mesh->partMask1Start = rangeStart;
                mesh->partMask1Count = count;
                break;
            case 2: mesh->partMask2Start = rangeStart;
                mesh->partMask2Count = count;
                break;
            case 3: mesh->partMask3Start = rangeStart;
                mesh->partMask3Count = count;
                break;
            default: break;
        }
    };

    int idxOffset = 0;
    for (const QuadRecord &rec: quadRecords) {
        if (static_cast<int>(rec.partMask) != currentPartMask) {
            closeRange(idxOffset);
            currentPartMask = rec.partMask;
            rangeStart = idxOffset;
        }
        uint32_t v = rec.vertexOffset;
        mesh->indexData[idxOffset + 0] = v + 0;
        mesh->indexData[idxOffset + 1] = v + 1;
        mesh->indexData[idxOffset + 2] = v + 2;
        mesh->indexData[idxOffset + 3] = v + 0;
        mesh->indexData[idxOffset + 4] = v + 2;
        mesh->indexData[idxOffset + 5] = v + 3;
        idxOffset += 6;
    }
    closeRange(idxOffset);

    if (outMeta != nullptr && env->GetArrayLength(outMeta) >= 9) {
        jint vals[9] = {
            mesh->vertexCount,
            mesh->indexCount,
            mesh->boneCount,
            mesh->partMask1Start, mesh->partMask1Count,
            mesh->partMask2Start, mesh->partMask2Count,
            mesh->partMask3Start, mesh->partMask3Count,
        };
        env->SetIntArrayRegion(outMeta, 0, 9, vals);
    }

    return reinterpret_cast<jlong>(mesh);
}

JNIEXPORT jobject JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nGetGpuMeshVertexBuffer(
    JNIEnv *env, jclass clazz, jlong handle) {
    auto *mesh = reinterpret_cast<NativeGpuMesh *>(handle);
    if (!mesh || !mesh->vertexData) return nullptr;
    return env->NewDirectByteBuffer(mesh->vertexData.get(),
                                    static_cast<jlong>(mesh->vertexCount) * sizeof(GpuVertex));
}

JNIEXPORT jobject JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nGetGpuMeshIndexBuffer(
    JNIEnv *env, jclass clazz, jlong handle) {
    auto *mesh = reinterpret_cast<NativeGpuMesh *>(handle);
    if (!mesh || !mesh->indexData) return nullptr;
    return env->NewDirectByteBuffer(mesh->indexData.get(),
                                    static_cast<jlong>(mesh->indexCount) * sizeof(uint32_t));
}

JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nReleaseGpuMeshScratch(
    JNIEnv *env, jclass clazz, jlong handle) {
    auto *mesh = reinterpret_cast<NativeGpuMesh *>(handle);
    if (!mesh) return;
    mesh->vertexData.reset();
    mesh->indexData.reset();
}

JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nFreeGpuMesh(
    JNIEnv *env, jclass clazz, jlong handle) {
    delete reinterpret_cast<NativeGpuMesh *>(handle);
}

JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nComputeBoneMatrices(
    JNIEnv *env, jclass clazz, jlong handle,
    jfloatArray rootPoseArr, jfloatArray rootNormalArr, jfloatArray animArray,
    jint packedLight, jobject outBoneBuffer) {
    auto *mesh = reinterpret_cast<NativeGpuMesh *>(handle);
    if (!mesh) return;

    auto *outRaw = static_cast<BoneDataOut *>(env->GetDirectBufferAddress(outBoneBuffer));
    if (!outRaw) return;

    jfloat *rootPose = static_cast<jfloat *>(env->GetPrimitiveArrayCritical(rootPoseArr, nullptr));
    jfloat *rootNormal = static_cast<jfloat *>(env->GetPrimitiveArrayCritical(rootNormalArr, nullptr));
    jfloat *anim = static_cast<jfloat *>(env->GetPrimitiveArrayCritical(animArray, nullptr));

    Mat4 rootPoseMat(rootPose);
    Mat4 rootNormalMat(UNINITIALIZED);
    rootNormalMat.m[0] = rootNormal[0];
    rootNormalMat.m[1] = rootNormal[1];
    rootNormalMat.m[2] = rootNormal[2];
    rootNormalMat.m[3] = 0.0f;
    rootNormalMat.m[4] = rootNormal[3];
    rootNormalMat.m[5] = rootNormal[4];
    rootNormalMat.m[6] = rootNormal[5];
    rootNormalMat.m[7] = 0.0f;
    rootNormalMat.m[8] = rootNormal[6];
    rootNormalMat.m[9] = rootNormal[7];
    rootNormalMat.m[10] = rootNormal[8];
    rootNormalMat.m[11] = 0.0f;
    rootNormalMat.m[12] = 0.0f;
    rootNormalMat.m[13] = 0.0f;
    rootNormalMat.m[14] = 0.0f;
    rootNormalMat.m[15] = 1.0f;

    const int glowLight = (15 << 4) | (15 << 20);
    std::fill(mesh->hiddenInherited.begin(), mesh->hiddenInherited.end(), 0);

    for (int bIdx: mesh->evalOrder) {
        const NativeBone &bone = mesh->bones[bIdx];

        int pOffset = bIdx * 12;
        float animRx = anim[pOffset + 0], animRy = anim[pOffset + 1], animRz = anim[pOffset + 2];
        float animTx = anim[pOffset + 3], animTy = anim[pOffset + 4], animTz = anim[pOffset + 5];
        float animSx = anim[pOffset + 6], animSy = anim[pOffset + 7], animSz = anim[pOffset + 8];
        float skipChildrenFlag = anim[pOffset + 10];

        float px = bone.pivotX * 0.0625f, py = bone.pivotY * 0.0625f, pz = bone.pivotZ * 0.0625f;
        float dx = px - animTx * 0.0625f;
        float dy = py + animTy * 0.0625f;
        float dz = pz + animTz * 0.0625f;

        float cx, sx, cy, sy, cz, sz;
        FAST_SINCOS(animRx, &sx, &cx);
        FAST_SINCOS(animRy, &sy, &cy);
        FAST_SINCOS(animRz, &sz, &cz);

        Mat4 localMat(UNINITIALIZED);
        localMat.m[0] = (cz * cy) * animSx;
        localMat.m[1] = (sz * cy) * animSx;
        localMat.m[2] = (-sy) * animSx;
        localMat.m[3] = 0.0f;
        localMat.m[4] = (cz * sy * sx - sz * cx) * animSy;
        localMat.m[5] = (sz * sy * sx + cz * cx) * animSy;
        localMat.m[6] = (cy * sx) * animSy;
        localMat.m[7] = 0.0f;
        localMat.m[8] = (cz * sy * cx + sz * sx) * animSz;
        localMat.m[9] = (sz * sy * cx - cz * sx) * animSz;
        localMat.m[10] = (cy * cx) * animSz;
        localMat.m[11] = 0.0f;
        localMat.m[12] = dx - (localMat.m[0] * px + localMat.m[4] * py + localMat.m[8] * pz);
        localMat.m[13] = dy - (localMat.m[1] * px + localMat.m[5] * py + localMat.m[9] * pz);
        localMat.m[14] = dz - (localMat.m[2] * px + localMat.m[6] * py + localMat.m[10] * pz);
        localMat.m[15] = 1.0f;

        bool inheritedHidden = (bone.parentIdx != -1) && mesh->hiddenInherited[bone.parentIdx] != 0;
        // Cycle defence (see nComputeModelVertices): parent not yet evaluated
        // in this pass -> fall back to the root matrix.
        const bool parentReady =
            bone.parentIdx != -1 && mesh->evalPos[bone.parentIdx] < mesh->evalPos[bIdx];
        // native-dll-round1: offset9 (HIDDEN) joins selfHidden so bone_skin.vsh
        // folds the hidden bone itself; inherited propagation unchanged, which
        // also hides the subtree exactly like the CPU visibleCache chain.
        bool selfHidden = inheritedHidden || (animSx == 0.0f || animSy == 0.0f || animSz == 0.0f)
                          || anim[pOffset + 9] != 0.0f;

        const Mat4 &parentGlobal =
            parentReady ? mesh->globalTransforms[bone.parentIdx] : rootPoseMat;
        Mat4 &globalMat = mesh->globalTransforms[bIdx];
        globalMat = parentGlobal;
        globalMat.mul(localMat);

        const Mat4 &parentNormal =
            parentReady ? mesh->globalNormals[bone.parentIdx] : rootNormalMat;
        Mat4 localNormalMat = localMat.normalMatrix4x4();
        Mat4 &globalNormalMat = mesh->globalNormals[bIdx];
        globalNormalMat = parentNormal;
        globalNormalMat.mul(localNormalMat);

        BoneDataOut &out = outRaw[bIdx];
        std::memcpy(out.transform, globalMat.m, 64);
        std::memcpy(out.normal, globalNormalMat.m, 64);
        out.packedLight = bone.glow ? glowLight : packedLight;
        out.isHidden = selfHidden ? 1 : 0;
        out.pad[0] = out.pad[1] = 0;

        mesh->hiddenInherited[bIdx] = (selfHidden || skipChildrenFlag != 0.0f) ? 1 : 0;
    }

    env->ReleasePrimitiveArrayCritical(rootPoseArr, rootPose, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(rootNormalArr, rootNormal, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(animArray, anim, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nComputeBoneMatricesLocal(
    JNIEnv *env, jclass clazz, jlong handle, jfloatArray animArray, jint packedLight,
    jobject outBoneBuffer) {
    auto *mesh = reinterpret_cast<NativeGpuMesh *>(handle);
    if (!mesh) return;

    auto *outRaw = static_cast<BoneDataOut *>(env->GetDirectBufferAddress(outBoneBuffer));
    if (!outRaw) return;

    jfloat *anim = static_cast<jfloat *>(env->GetPrimitiveArrayCritical(animArray, nullptr));

    const int glowLight = (15 << 4) | (15 << 20);
    std::fill(mesh->hiddenInherited.begin(), mesh->hiddenInherited.end(), 0);

    for (int bIdx: mesh->evalOrder) {
        const NativeBone &bone = mesh->bones[bIdx];

        int pOffset = bIdx * 12;
        float animRx = anim[pOffset + 0], animRy = anim[pOffset + 1], animRz = anim[pOffset + 2];
        float animTx = anim[pOffset + 3], animTy = anim[pOffset + 4], animTz = anim[pOffset + 5];
        float animSx = anim[pOffset + 6], animSy = anim[pOffset + 7], animSz = anim[pOffset + 8];
        float skipChildrenFlag = anim[pOffset + 10];

        float px = bone.pivotX * 0.0625f, py = bone.pivotY * 0.0625f, pz = bone.pivotZ * 0.0625f;
        float dx = px - animTx * 0.0625f;
        float dy = py + animTy * 0.0625f;
        float dz = pz + animTz * 0.0625f;

        float cx, sx, cy, sy, cz, sz;
        FAST_SINCOS(animRx, &sx, &cx);
        FAST_SINCOS(animRy, &sy, &cy);
        FAST_SINCOS(animRz, &sz, &cz);

        Mat4 localMat(UNINITIALIZED);
        localMat.m[0] = (cz * cy) * animSx;
        localMat.m[1] = (sz * cy) * animSx;
        localMat.m[2] = (-sy) * animSx;
        localMat.m[3] = 0.0f;
        localMat.m[4] = (cz * sy * sx - sz * cx) * animSy;
        localMat.m[5] = (sz * sy * sx + cz * cx) * animSy;
        localMat.m[6] = (cy * sx) * animSy;
        localMat.m[7] = 0.0f;
        localMat.m[8] = (cz * sy * cx + sz * sx) * animSz;
        localMat.m[9] = (sz * sy * cx - cz * sx) * animSz;
        localMat.m[10] = (cy * cx) * animSz;
        localMat.m[11] = 0.0f;
        localMat.m[12] = dx - (localMat.m[0] * px + localMat.m[4] * py + localMat.m[8] * pz);
        localMat.m[13] = dy - (localMat.m[1] * px + localMat.m[5] * py + localMat.m[9] * pz);
        localMat.m[14] = dz - (localMat.m[2] * px + localMat.m[6] * py + localMat.m[10] * pz);
        localMat.m[15] = 1.0f;

        bool inheritedHidden = (bone.parentIdx != -1) && mesh->hiddenInherited[bone.parentIdx] != 0;
        // Cycle defence + offset9, same as nComputeBoneMatrices.
        const bool parentReady =
            bone.parentIdx != -1 && mesh->evalPos[bone.parentIdx] < mesh->evalPos[bIdx];
        bool selfHidden = inheritedHidden || (animSx == 0.0f || animSy == 0.0f || animSz == 0.0f)
                          || anim[pOffset + 9] != 0.0f;

        Mat4 &globalMat = mesh->globalTransforms[bIdx];
        Mat4 localNormalMat = localMat.normalMatrix4x4();
        Mat4 &globalNormalMat = mesh->globalNormals[bIdx];

        if (parentReady) {
            globalMat = mesh->globalTransforms[bone.parentIdx];
            globalMat.mul(localMat);

            globalNormalMat = mesh->globalNormals[bone.parentIdx];
            globalNormalMat.mul(localNormalMat);
        } else {
            globalMat = localMat;
            globalNormalMat = localNormalMat;
        }

        BoneDataOut &out = outRaw[bIdx];
        std::memcpy(out.transform, globalMat.m, 64);
        std::memcpy(out.normal, globalNormalMat.m, 64);
        out.packedLight = bone.glow ? glowLight : packedLight;
        out.isHidden = selfHidden ? 1 : 0;
        out.pad[0] = out.pad[1] = 0;

        mesh->hiddenInherited[bIdx] = (selfHidden || skipChildrenFlag != 0.0f) ? 1 : 0;
    }

    env->ReleasePrimitiveArrayCritical(animArray, anim, JNI_ABORT);
}

static const JNINativeMethod gMethods[] = {
    {
        (char *) "nInitModelCache", (char *) "(Ljava/nio/ByteBuffer;)J",
        reinterpret_cast<void *>(Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nInitModelCache)
    },
    {
        (char *) "nDestroyModelCache", (char *) "(J)V",
        reinterpret_cast<void *>(Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nDestroyModelCache)
    },
    {
        // [B] stateArray inserted after animArray.
        (char *) "nComputeModelVertices", (char *) "(JLjava/lang/Object;[F[F[FIIIFFFF)V",
        reinterpret_cast<void *>(
            Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nComputeModelVertices)
    },
    {
        (char *) "nBuildGpuMesh", (char *) "(Ljava/nio/ByteBuffer;[I)J",
        reinterpret_cast<void *>(Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nBuildGpuMesh)
    },
    {
        (char *) "nGetGpuMeshVertexBuffer", (char *) "(J)Ljava/nio/ByteBuffer;",
        reinterpret_cast<void *>(
            Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nGetGpuMeshVertexBuffer)
    },
    {
        (char *) "nGetGpuMeshIndexBuffer", (char *) "(J)Ljava/nio/ByteBuffer;",
        reinterpret_cast<void *>(
            Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nGetGpuMeshIndexBuffer)
    },
    {
        (char *) "nReleaseGpuMeshScratch", (char *) "(J)V",
        reinterpret_cast<void *>(
            Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nReleaseGpuMeshScratch)
    },
    {
        (char *) "nFreeGpuMesh", (char *) "(J)V",
        reinterpret_cast<void *>(Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nFreeGpuMesh)
    },
    {
        (char *) "nComputeBoneMatrices", (char *) "(J[F[F[FILjava/nio/ByteBuffer;)V",
        reinterpret_cast<void *>(
            Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nComputeBoneMatrices)
    },
    {
        (char *) "nComputeBoneMatricesLocal", (char *) "(J[FILjava/nio/ByteBuffer;)V",
        reinterpret_cast<void *>(
            Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nComputeBoneMatricesLocal)
    },
    {
        // [A] 11th export.
        (char *) "nInitSIMD",
        (char *) "(Ljava/lang/Class;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
                 "Ljava/lang/String;Ljava/lang/String;Ljava/lang/Class;)V",
        reinterpret_cast<void *>(
            Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nInitSIMD)
    },
};

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
__attribute__((target("no-avx2,no-fma")))
static bool hasRequiredCpuSupport() {
    unsigned int eax, ebx, ecx, edx;

    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) return false;
    if (eax < 7) return false;

    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    if (!(ecx & (1u << 12))) return false;
    if (!(ecx & (1u << 27))) return false;
    if (!(ecx & (1u << 28))) return false;

    unsigned int xcr0_lo, xcr0_hi;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    if ((xcr0_lo & 0x6) != 0x6) return false;

    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    if (!(ebx & (1u << 5))) return false;

    return true;
}
#else
static bool hasRequiredCpuSupport() { return true; }
#endif

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    if (!hasRequiredCpuSupport()) {
        return JNI_ERR;
    }

    jclass clazzModel = env->FindClass("com/elfmcys/yesstevemodel/geckolib3/geo/render/built/GeoModel");
    if (clazzModel == nullptr) return JNI_ERR;
    if (env->RegisterNatives(clazzModel, gMethods, sizeof(gMethods) / sizeof(gMethods[0])) < 0)
        return JNI_ERR;

    jclass clazzRenderer = env->FindClass("com/elfmcys/yesstevemodel/geckolib3/geo/NativeModelRenderer");
    if (clazzRenderer != nullptr) {
        g_NativeModelRendererClass = (jclass) env->NewGlobalRef(clazzRenderer);
        g_submitVerticesID = env->GetStaticMethodID(
            g_NativeModelRendererClass, "submitVertices",
            "(Ljava/lang/Object;ILjava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V");
    }

    return JNI_VERSION_1_6;
}
} // extern "C"
