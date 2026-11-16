# Déploiement Android : QNN (Hexagon NPU) ou GPU

Le modèle exporté est un ONNX **NCHW fp32**, 2 entrées `frame0`/`frame1`
(1×3×256×448), 1 sortie `frame_mid` (1×3×256×448), valeurs dans `[0,1]`.
Trois chemins, du meilleur au plus simple.

---

## A. QNN via Qualcomm AI Hub (recommandé, aucune bidouille SDK)

AI Hub compile le modèle pour **Hexagon HTP** (fp16 par défaut) et te renvoie un
contexte QNN + un bundle Android exécutable, et peut benchmarker sur un vrai
Snapdragon.

```bash
pip install qai-hub
qai-hub configure --api_token "$QAI_HUB_API_TOKEN"   # token: app.aihub.qualcomm.com

python - <<'PY'
import qai_hub as hub
m = hub.upload_model("checkpoints/mobileifnet_fp16.onnx")
for d in ["Samsung Galaxy S23", "Samsung Galaxy S24", "Xiaomi 13"]:
    dev = hub.Device(d)
    job = hub.submit_compile_job(
        model=m,
        device=dev,
        input_specs={"frame0": ((1, 3, 256, 448), "float32"),
                     "frame1": ((1, 3, 256, 448), "float32")},
        options="--target_runtime qnn_context_binary",
    )
    target = job.get_target_model()
    print(d, target.download("checkpoints/"))
    prof = hub.submit_profile_job(model=target, device=dev)
    print(prof.download_job_results() if hasattr(prof, "download_job_results") else prof)
PY
```

Puis sur le téléphone :

```bash
# QNN SDK (ai-engine-direct / qnn-net-run), pousser le contexte + libs
adb push mobileifnet_qnn.bin /data/local/tmp/
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./qnn-net-run \
  --backend libQnnHtp.so --model mobileifnet_qnn.bin \
  --input_list input_list.txt"
```

Attendu (ordre de grandeur, à confirmer par profil AI Hub) : **5–15 ms par
frame 256×448** sur un HTP récent → 60–200 fps, soit largement de quoi faire du
**2x, 4x, 8x en temps réel** si tu ne réutilises pas les frames.

> Le `GridSample` (warp bilinéaire) est supporté par QNN en fp16 ; si un
> opérateur tombe en fallback CPU, le profil AI Hub le montre (colonne
> `layer_type` / `backend`) : dans ce cas, passe le modèle en int8 avec QAT plutôt
> que de bricoler l'opérateur.

---

## B. onnxruntime + QNN Execution Provider (dans ton app)

Avec le SDK QNN (`libQnnHtp.so`, `libQnnHtpV75Skel.so`, …) dans `jniLibs/`,
et `onnxruntime-android` + le package `onnxruntime-qnn` :

```kotlin
val so = OrtSession.SessionOptions()
so.addConfigEntry("session.use_qnn", "1")
so.addConfigEntry("qnn.backend_path", "libQnnHtp.so")
so.addConfigEntry("qnn.hardware_backend", "HTP")
val session = OrtEnvironment.getEnvironment().createSession(modelBytes, so)
```

Sans QNN, tu tombes automatiquement sur CPU/XNNPACK :
`so.addNnapi()` (NNAPI) ou `so.addXnnpack(mapOf("intra_op_num_threads" to "4"))`.

**GPU (plus simple, portable)** : `so.addConfigEntry("session.use_nnapi", "1")`
ou côté TFLite :

```bash
pip install onnx2tf tensorflow
onnx2tf -i checkpoints/mobileifnet_fp16.onnx -o tflite/   # -> .tflite
```

puis Android :

```kotlin
val opts = Interpreter.Options().apply {
    addDelegate(GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(true)))
}
val interp = Interpreter(FileUtil.loadMappedFile(ctx, "mobileifnet.tflite"), opts)
```

Le GPU mobile n'aime pas `GridSample` : si `onnx2tf` échoue sur cet opérateur,
utilise la version int8 côté CPU (4 threads) ou reste sur QNN.

---

## C. Contrat d'entrée / prétraitement (identique partout)

```
frame -> RGB float32 / 255 -> (tensor 1x3xHxW, NCHW)
paires:  (I[t], I[t+1]) -> frame interpolée à t+0.5
  2x : insérer une frame entre chaque paire
  4x : récurser (I[t], mid) et (mid, I[t+1])
```

Pré-traitement le plus rapide sur Android : `ImageReader` en `YUV_420_888` →
conversion NV21 → `float[3][H][W]` (pas de Bitmap), ou `AHardwareBuffer` +
`GL_TEXTURE_EXTERNAL_OES` si tu veux rester 100 % GPU.

Recommandations perf téléphone :

* batch = 1, résolution 256×448 (ou 128×224 pour du 4x temps réel) ;
* garder les mêmes buffers d'entrée/sortie (zero-copy) ;
* QNN fp16 > int8 sur le HTP pour la qualité (le warp est sensible au bruit) ;
* mesurer avec `adb shell dumpsys SurfaceFlinger --latency` ou AI Hub profile,
  pas avec des fps « ressentis ».
