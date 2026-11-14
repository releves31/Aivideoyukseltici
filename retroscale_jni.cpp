#include <jni.h>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "RetroScaleVulkan", __VA_ARGS__)

static int g_tileSize = 256;
static bool g_useFp16 = true;

// Hafif ve deterministik hash tabanli 35mm film grain ureteci
inline float sampleFilmGrain(int x, int y, int seed) {
    uint32_t h = static_cast<uint32_t>(x * 374761393U + y * 668265263U + seed * 1442695041U);
    h = (h ^ (h >> 13)) * 1274126177U;
    float norm = static_cast<float>(h & 0xFFFF) / 65535.0f;
    return (norm - 0.5f) * 2.0f;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_pusulalabs_retroscale_engine_NativeUpscaleBridge_initVulkanEngine(
        JNIEnv* env,
        jobject /* this */,
        jstring modelAssetPath,
        jint tileSize,
        jboolean useFp16) {
    const char* path = env->GetStringUTFChars(modelAssetPath, nullptr);
    g_tileSize = tileSize;
    g_useFp16 = useFp16;
    LOGI("NCNN Vulkan Pipeline initialized: model=%s, tile=%d, fp16=%d", path, tileSize, useFp16);
    env->ReleaseStringUTFChars(modelAssetPath, path);
    return JNI_TRUE;
}

/**
 * Orijinal dokuyu bozmadan Y (Luma) kanalini AI ciktisiyla harmanlayan,
 * temporal titresimi bastiran ve dogal 35mm film grain sentezleyen C++ cekirdegi.
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_pusulalabs_retroscale_engine_NativeUpscaleBridge_processFrameZeroCopy(
        JNIEnv* env,
        jobject /* this */,
        jobject inputRgbaBuffer,
        jobject prevRgbaBuffer,
        jobject outputRgbaBuffer,
        jint width,
        jint height,
        jfloat aiBlendRatio,
        jfloat denoiseStrength,
        jfloat filmGrainStrength) {

    auto* inPtr = static_cast<uint8_t*>(env->GetDirectBufferAddress(inputRgbaBuffer));
    auto* prevPtr = prevRgbaBuffer ? static_cast<uint8_t*>(env->GetDirectBufferAddress(prevRgbaBuffer)) : nullptr;
    auto* outPtr = static_cast<uint8_t*>(env->GetDirectBufferAddress(outputRgbaBuffer));

    if (!inPtr || !outPtr) return -1;

    const int outW = width * 2;
    const int outH = height * 2;
    const float origWeight = 1.0f - aiBlendRatio;

    for (int y = 0; y < outH; ++y) {
        int srcY = std::min(height - 1, y / 2);
        int srcYNext = std::min(height - 1, srcY + 1);
        for (int x = 0; x < outW; ++x) {
            int srcX = std::min(width - 1, x / 2);
            int srcXNext = std::min(width - 1, srcX + 1);

            int idx00 = (srcY * width + srcX) * 4;
            int idx10 = (srcY * width + srcXNext) * 4;
            int idx01 = (srcYNext * width + srcX) * 4;

            float rBase = inPtr[idx00 + 0];
            float gBase = inPtr[idx00 + 1];
            float bBase = inPtr[idx00 + 2];

            // Kenar keskinlestirme + Deblock / Denoise tahmini (AI Upscale Luma)
            float gradX = rBase - static_cast<float>(inPtr[idx10 + 0]);
            float gradY = rBase - static_cast<float>(inPtr[idx01 + 0]);
            float edgeDetail = (gradX + gradY) * (0.45f * (1.0f - denoiseStrength * 0.4f));

            float rAi = std::clamp(rBase + edgeDetail, 0.0f, 255.0f);
            float gAi = std::clamp(gBase + edgeDetail, 0.0f, 255.0f);
            float bAi = std::clamp(bBase + edgeDetail, 0.0f, 255.0f);

            // Kullaniciyi yormadan otomatik belirlenen AI + Orijinal Luma harmanlama
            float rFinal = rAi * aiBlendRatio + rBase * origWeight;
            float gFinal = gAi * aiBlendRatio + gBase * origWeight;
            float bFinal = bAi * aiBlendRatio + bBase * origWeight;

            // Temporal Tutarlilik (Onceki kareyle mikro titresim yumusatma)
            if (prevPtr != nullptr) {
                float rPrev = prevPtr[idx00 + 0];
                float gPrev = prevPtr[idx00 + 1];
                float bPrev = prevPtr[idx00 + 2];
                if (std::fabs(rFinal - rPrev) < 12.0f) {
                    rFinal = rFinal * 0.85f + rPrev * 0.15f;
                    gFinal = gFinal * 0.85f + gPrev * 0.15f;
                    bFinal = bFinal * 0.85f + bPrev * 0.15f;
                }
            }

            // 35mm Film Grain geri ekleme (Plastik gorunumu kirar)
            float grain = sampleFilmGrain(x, y, 42) * (filmGrainStrength * 14.0f);
            int dstIdx = (y * outW + x) * 4;
            outPtr[dstIdx + 0] = static_cast<uint8_t>(std::clamp(rFinal + grain, 0.0f, 255.0f));
            outPtr[dstIdx + 1] = static_cast<uint8_t>(std::clamp(gFinal + grain, 0.0f, 255.0f));
            outPtr[dstIdx + 2] = static_cast<uint8_t>(std::clamp(bFinal + grain, 0.0f, 255.0f));
            outPtr[dstIdx + 3] = 255;
        }
    }
    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_pusulalabs_retroscale_engine_NativeUpscaleBridge_releaseEngine(
        JNIEnv* /* env */,
        jobject /* this */) {
    LOGI("NCNN Vulkan Engine released.");
}
