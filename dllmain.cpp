#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC optimize("O3,unroll-loops")
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC target("sse4.1,fma")
  #endif
  #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
  #define SSE2NEON_SUPPRESS_WARNINGS
  #include "sse2neon.h"
#else
  #include <immintrin.h>
#endif

#include <vector>
#include <cmath>
#include <cstring>
#include <functional>
#include "jni.h"
#include <cstdint>

#if defined(__GLIBC__) || defined(__BIONIC__)
#define FAST_SINCOS(x, s, c) sincosf((x), (s), (c))
#else
inline void ms_sincosf(float x, float* s, float* c) {
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
    Mat4(Uninitialized) {}
    Mat4(const float* data) { std::memcpy(m, data, 16 * sizeof(float)); }

    inline void identity() {
        std::memset(m, 0, 16 * sizeof(float));
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    inline void mul(const Mat4& right) {
        __m128 l0 = _mm_load_ps(&m[0]);
        __m128 l1 = _mm_load_ps(&m[4]);
        __m128 l2 = _mm_load_ps(&m[8]);
        __m128 l3 = _mm_load_ps(&m[12]);

        auto mac = [&](const float* r_col) {
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
        res.m[0] = m[0]; res.m[1] = m[1]; res.m[2] = m[2];  res.m[3] = 0.0f;
        res.m[4] = m[4]; res.m[5] = m[5]; res.m[6] = m[6];  res.m[7] = 0.0f;
        res.m[8] = m[8]; res.m[9] = m[9]; res.m[10] = m[10]; res.m[11] = 0.0f;
        res.m[12] = 0.0f; res.m[13] = 0.0f; res.m[14] = 0.0f; res.m[15] = 1.0f;
        return res;
    }
};

struct NativeModel {
    std::vector<NativeBone> bones;
    std::vector<FastQuad> fastQuads;
    std::vector<int> evalOrder;

    std::vector<Mat4> cacheGlobalTransforms;
    std::vector<Mat4> cacheGlobalNormals;
    std::vector<PrecomputedBoneMats> cachePrecompMats;
    std::vector<int> visibleBones;

    jfloatArray cachedFloatArray = nullptr;
    int cachedFloatCapacity = 0;

    jintArray cachedIntArray = nullptr;
    int cachedIntCapacity = 0;
};

extern "C" {

JNIEXPORT jlong JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nInitModelCache(
    JNIEnv* env, jclass clazz, jobject buffer) {
    char* data = (char*)env->GetDirectBufferAddress(buffer);
    if (!data) return 0;

    NativeModel* model = new NativeModel();
    int offset = 0;

    auto readInt = [&]() { int v; std::memcpy(&v, data + offset, 4); offset += 4; return v; };
    auto readFloat = [&]() { float v; std::memcpy(&v, data + offset, 4); offset += 4; return v; };
    auto readByte = [&]() { char v = data[offset]; offset += 1; return v; };

    int boneCount = readInt();
    model->bones.resize(boneCount);
    model->cacheGlobalTransforms.resize(boneCount);
    model->cacheGlobalNormals.resize(boneCount);
    model->cachePrecompMats.resize(boneCount);
    model->visibleBones.reserve(boneCount);
    model->fastQuads.reserve(boneCount * 20);

    std::vector<std::vector<int>> children(boneCount);

    for (int i = 0; i < boneCount; ++i) {
        NativeBone& bone = model->bones[i];
        bone.parentIdx = readInt();
        if (bone.parentIdx != -1) {
            children[bone.parentIdx].push_back(i);
        }

        bone.partMask = readInt();
        bone.glow = readByte() != 0;
        bone.pivotX = readFloat();
        bone.pivotY = readFloat();
        bone.pivotZ = readFloat();

        bone.quadStart = model->fastQuads.size();

        int cubeCount = readInt();
        for (int j = 0; j < cubeCount; ++j) {
            bool cullable = readByte() != 0;
            int quadCount = readInt();
            for (int k = 0; k < quadCount; ++k) {
                FastQuad fq;
                fq.boneIdx = i;
                fq.cullable = cullable;

                alignas(16) float tmpX[4], tmpY[4], tmpZ[4], tmpU[4], tmpV[4];
                for (int v = 0; v < 4; ++v) {
                    tmpX[v] = readFloat(); tmpY[v] = readFloat(); tmpZ[v] = readFloat();
                }
                for (int v = 0; v < 4; ++v) {
                    tmpU[v] = readFloat(); tmpV[v] = readFloat();
                }
                fq.nx = readFloat(); fq.ny = readFloat(); fq.nz = readFloat();

                fq.x = _mm_load_ps(tmpX); fq.y = _mm_load_ps(tmpY); fq.z = _mm_load_ps(tmpZ);
                fq.u = _mm_load_ps(tmpU); fq.v = _mm_load_ps(tmpV);
                model->fastQuads.push_back(fq);
            }
        }
        bone.quadCount = model->fastQuads.size() - bone.quadStart;
    }
    model->fastQuads.shrink_to_fit();

    std::function<int(int)> dfs = [&](int idx) -> int {
        model->evalOrder.push_back(idx);
        int count = 0;
        for (int child : children[idx]) {
            count += dfs(child);
        }
        model->bones[idx].subtreeCount = count;
        return count + 1;
        };

    for (int i = 0; i < boneCount; ++i) {
        if (model->bones[i].parentIdx == -1) dfs(i);
    }

    return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nDestroyModelCache(
    JNIEnv* env, jclass clazz, jlong handle) {
    delete reinterpret_cast<NativeModel*>(handle);
}

JNIEXPORT void JNICALL Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nComputeModelVertices(
    JNIEnv* env, jclass clazz, jlong handle, jobject vertexConsumer,
    jfloatArray matrixArray, jfloatArray animArray,
    jint renderPartMask, jint packedLight, jint packedOverlay,
    jfloat r, jfloat g, jfloat b, jfloat a) {

    NativeModel* model = reinterpret_cast<NativeModel*>(handle);
    if (!model || model->fastQuads.empty()) return;

    jfloat* matricesData = env->GetFloatArrayElements(matrixArray, nullptr);
    jfloat* animData = env->GetFloatArrayElements(animArray, nullptr);

    Mat4 rootPoseMat(matricesData);
    float* rootNormalArr = matricesData + 16;
    Mat4 projMat(matricesData + 32);

    Mat4 rootNormalMat(UNINITIALIZED);
    rootNormalMat.m[0] = rootNormalArr[0]; rootNormalMat.m[1] = rootNormalArr[1]; rootNormalMat.m[2] = rootNormalArr[2]; rootNormalMat.m[3] = 0.0f;
    rootNormalMat.m[4] = rootNormalArr[3]; rootNormalMat.m[5] = rootNormalArr[4]; rootNormalMat.m[6] = rootNormalArr[5]; rootNormalMat.m[7] = 0.0f;
    rootNormalMat.m[8] = rootNormalArr[6]; rootNormalMat.m[9] = rootNormalArr[7]; rootNormalMat.m[10] = rootNormalArr[8]; rootNormalMat.m[11] = 0.0f;
    rootNormalMat.m[12] = 0.0f; rootNormalMat.m[13] = 0.0f; rootNormalMat.m[14] = 0.0f; rootNormalMat.m[15] = 1.0f;

    int glowLight = (15 << 4) | (15 << 20);
    size_t boneCount = model->bones.size();

    model->visibleBones.clear();

    int k = 0;
    while (k < boneCount) {
        int bIdx = model->evalOrder[k];
        NativeBone& bone = model->bones[bIdx];

        int pOffset = bIdx * 12;
        float animRx = animData[pOffset + 0], animRy = animData[pOffset + 1], animRz = animData[pOffset + 2];
        float animTx = animData[pOffset + 3], animTy = animData[pOffset + 4], animTz = animData[pOffset + 5];
        float animSx = animData[pOffset + 6], animSy = animData[pOffset + 7], animSz = animData[pOffset + 8];
        float skipChildrenFlag = animData[pOffset + 10];

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

        Mat4 parentGlobal = (bone.parentIdx != -1) ? model->cacheGlobalTransforms[bone.parentIdx] : rootPoseMat;
        Mat4 globalMat = parentGlobal;
        globalMat.mul(localMat);
        model->cacheGlobalTransforms[bIdx] = globalMat;

        Mat4 parentNormal = (bone.parentIdx != -1) ? model->cacheGlobalNormals[bone.parentIdx] : rootNormalMat;
        Mat4 localNormalMat = localMat.normalMatrix4x4();
        Mat4 globalNormalMat = parentNormal;
        globalNormalMat.mul(localNormalMat);
        model->cacheGlobalNormals[bIdx] = globalNormalMat;

        auto& precomp = model->cachePrecompMats[bIdx];
        std::memcpy(precomp.gb, globalMat.m, 16 * sizeof(float));

        precomp.gn_c0 = _mm_load_ps(&globalNormalMat.m[0]);
        precomp.gn_c1 = _mm_load_ps(&globalNormalMat.m[4]);
        precomp.gn_c2 = _mm_load_ps(&globalNormalMat.m[8]);
        precomp.currentLight = bone.glow ? glowLight : packedLight;

        if (animSx == 0.0f || animSy == 0.0f || animSz == 0.0f) {
            k += bone.subtreeCount + 1;
            continue;
        }

        model->visibleBones.push_back(bIdx);

        if (skipChildrenFlag != 0.0f) {
            k += bone.subtreeCount + 1;
            continue;
        }
        k++;
    }

    int maxVertices = 0;
    for (int bIdx : model->visibleBones) {
        const NativeBone& bone = model->bones[bIdx];
        if (bone.quadCount == 0) continue;
        if (renderPartMask != 0 && bone.partMask != renderPartMask && bone.partMask != 3) continue;
        maxVertices += bone.quadCount * 4;
    }

    if (maxVertices == 0) {
        env->ReleaseFloatArrayElements(matrixArray, matricesData, JNI_ABORT);
        env->ReleaseFloatArrayElements(animArray, animData, JNI_ABORT);
        return;
    }

    int maxFloats = maxVertices * 12;
    int maxInts = maxVertices * 2;

    static thread_local std::vector<float> fData;
    static thread_local std::vector<int> iData;

    fData.reserve(maxFloats);
    iData.reserve(maxInts);
    
    float* fPtr = fData.data();
    int* iPtr = iData.data();

    int actualVertices = 0;

    __m128 p00 = _mm_set1_ps(projMat.m[0]), p01 = _mm_set1_ps(projMat.m[4]), p02 = _mm_set1_ps(projMat.m[8]), p03 = _mm_set1_ps(projMat.m[12]);
    __m128 p10 = _mm_set1_ps(projMat.m[1]), p11 = _mm_set1_ps(projMat.m[5]), p12 = _mm_set1_ps(projMat.m[9]), p13 = _mm_set1_ps(projMat.m[13]);
    __m128 p30 = _mm_set1_ps(projMat.m[3]), p31 = _mm_set1_ps(projMat.m[7]), p32 = _mm_set1_ps(projMat.m[11]), p33 = _mm_set1_ps(projMat.m[15]);

    for (int bIdx : model->visibleBones) {
        const NativeBone& bone = model->bones[bIdx];
        if (bone.quadCount == 0) continue;
        if (renderPartMask != 0 && bone.partMask != renderPartMask && bone.partMask != 3) continue;

        const auto& pMat = model->cachePrecompMats[bIdx];

        __m128 gb0 = _mm_set1_ps(pMat.gb[0]), gb1 = _mm_set1_ps(pMat.gb[1]), gb2 = _mm_set1_ps(pMat.gb[2]);
        __m128 gb4 = _mm_set1_ps(pMat.gb[4]), gb5 = _mm_set1_ps(pMat.gb[5]), gb6 = _mm_set1_ps(pMat.gb[6]);
        __m128 gb8 = _mm_set1_ps(pMat.gb[8]), gb9 = _mm_set1_ps(pMat.gb[9]), gb10 = _mm_set1_ps(pMat.gb[10]);
        __m128 gb12 = _mm_set1_ps(pMat.gb[12]), gb13 = _mm_set1_ps(pMat.gb[13]), gb14 = _mm_set1_ps(pMat.gb[14]);

        for (int q = 0; q < bone.quadCount; ++q) {
            const FastQuad& fq = model->fastQuads[bone.quadStart + q];

            __m128 gX = MADD_PS(gb0, fq.x, MADD_PS(gb4, fq.y, MADD_PS(gb8, fq.z, gb12)));
            __m128 gY = MADD_PS(gb1, fq.x, MADD_PS(gb5, fq.y, MADD_PS(gb9, fq.z, gb13)));
            __m128 gZ = MADD_PS(gb2, fq.x, MADD_PS(gb6, fq.y, MADD_PS(gb10, fq.z, gb14)));

            if (fq.cullable) {
                __m128 pX = MADD_PS(p00, gX, MADD_PS(p01, gY, MADD_PS(p02, gZ, p03)));
                __m128 pY = MADD_PS(p10, gX, MADD_PS(p11, gY, MADD_PS(p12, gZ, p13)));
                __m128 pW = MADD_PS(p30, gX, MADD_PS(p31, gY, MADD_PS(p32, gZ, p33)));

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
                MADD_PS(pMat.gn_c1, _mm_set1_ps(fq.ny), _mm_mul_ps(pMat.gn_c2, _mm_set1_ps(fq.nz))));
            __m128 dp = _mm_mul_ps(n_res, n_res);
            __m128 sum = _mm_add_ps(dp, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(2, 3, 0, 1)));
            sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
            n_res = _mm_mul_ps(n_res, _mm_rsqrt_ps(_mm_max_ps(sum, _mm_set1_ps(1e-8f))));

            alignas(16) float finalNorm[4];
            _mm_store_ps(finalNorm, n_res);

            alignas(16) float fx[4], fy[4], fz[4], fu[4], fv[4];
            _mm_store_ps(fx, gX); _mm_store_ps(fy, gY); _mm_store_ps(fz, gZ);
            _mm_store_ps(fu, fq.u); _mm_store_ps(fv, fq.v);


            for (int v = 0; v < 4; ++v) {
                *fPtr++ = fx[v];
                *fPtr++ = fy[v];
                *fPtr++ = fz[v];
                *fPtr++ = r;
                *fPtr++ = g;
                *fPtr++ = b;
                *fPtr++ = a;
                *fPtr++ = fu[v];
                *fPtr++ = fv[v];
                *fPtr++ = finalNorm[0];
                *fPtr++ = finalNorm[1];
                *fPtr++ = finalNorm[2];

                *iPtr++ = packedOverlay;
                *iPtr++ = pMat.currentLight;
            }

            actualVertices += 4;
        }
    }

    env->ReleaseFloatArrayElements(matrixArray, matricesData, JNI_ABORT);
    env->ReleaseFloatArrayElements(animArray, animData, JNI_ABORT);

    if (actualVertices > 0 && g_NativeModelRendererClass && g_submitVerticesID) {
        int actualFloats = actualVertices * 12;
        int actualInts = actualVertices * 2;

        if (actualFloats > model->cachedFloatCapacity || model->cachedFloatArray == nullptr) {
            if (model->cachedFloatArray) {
                env->DeleteGlobalRef(model->cachedFloatArray);
            }
            int newCap = actualFloats + (actualFloats / 5) + 1200;
            jfloatArray localF = env->NewFloatArray(newCap);
            model->cachedFloatArray = (jfloatArray)env->NewGlobalRef(localF);
            model->cachedFloatCapacity = newCap;
            env->DeleteLocalRef(localF);
        }

        if (actualInts > model->cachedIntCapacity || model->cachedIntArray == nullptr) {
            if (model->cachedIntArray) {
                env->DeleteGlobalRef(model->cachedIntArray);
            }
            int newCap = actualInts + (actualInts / 5) + 200;
            jintArray localI = env->NewIntArray(newCap);
            model->cachedIntArray = (jintArray)env->NewGlobalRef(localI);
            model->cachedIntCapacity = newCap;
            env->DeleteLocalRef(localI);
        }

        env->SetFloatArrayRegion(model->cachedFloatArray, 0, actualFloats, fData.data());
        env->SetIntArrayRegion(model->cachedIntArray, 0, actualInts, reinterpret_cast<const jint*>(iData.data()));

        env->CallStaticVoidMethod(g_NativeModelRendererClass, g_submitVerticesID, vertexConsumer, actualVertices, model->cachedFloatArray, model->cachedIntArray);
    }
}

static const JNINativeMethod gMethods[] = {
    { (char*)"nInitModelCache", (char*)"(Ljava/nio/ByteBuffer;)J", reinterpret_cast<void*>(Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nInitModelCache) },
    { (char*)"nDestroyModelCache", (char*)"(J)V", reinterpret_cast<void*>(Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nDestroyModelCache) },
    { (char*)"nComputeModelVertices", (char*)"(JLjava/lang/Object;[F[FIIIFFFF)V", reinterpret_cast<void*>(Java_com_elfmcys_yesstevemodel_geckolib3_geo_render_built_GeoModel_nComputeModelVertices) }
};

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    jclass clazzModel = env->FindClass("com/elfmcys/yesstevemodel/geckolib3/geo/render/built/GeoModel");
    if (clazzModel == nullptr) return JNI_ERR;
    if (env->RegisterNatives(clazzModel, gMethods, 3) < 0) return JNI_ERR;

    jclass clazzRenderer = env->FindClass("com/elfmcys/yesstevemodel/geckolib3/geo/NativeModelRenderer");
    if (clazzRenderer != nullptr) {
        g_NativeModelRendererClass = (jclass)env->NewGlobalRef(clazzRenderer);
        g_submitVerticesID = env->GetStaticMethodID(g_NativeModelRendererClass, "submitVertices", "(Ljava/lang/Object;I[F[I)V");
    }

    return JNI_VERSION_1_6;
}

} // extern "C"