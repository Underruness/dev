# HANDOFF — reprendre dans une nouvelle discussion

Dépôt GitHub : **https://github.com/Underruness/dev**
Tout le code, les docs et les scripts y sont. Les poids entraînés restent dans le
workspace de ce chat (`checkpoints/`), voir §5 pour les pousser.

## 1. État exact (mesuré, pas estimé)

| Élément | État |
|---|---|
| Modèle `MobileIFNet` — 137 087 params, ONNX propre (vérifié) | ✅ |
| Entraînement réel sur la VM CPU : **3 549 itérations** (~26 min) | ✅ |
| Dataset : 7 clips Xiph/DERF 720p → **2 086 triplets** consécutifs | ✅ |
| Validation sur clips tenus à l'écart (`Johnny` + `shields`) | ✅ |
| **39,41 dB PSNR / 0,970 SSIM** vs **33,51 dB** (mélange linéaire) | ✅ |
| Export ONNX + int8 QDQ + fp16 | ✅ |
| Guide QNN (AI Hub) / ORT-QNN-EP / TFLite GPU | ✅ `android/README.md` |
| Mesures de latence sur téléphone | ❌ aucun appareil dans la VM |
| Modèle SOTA | ❌ impossible sans GPU (voir §2) |

## 2. À dire à l'utilisateur en premier

Cette VM **n'a pas de GPU** et **se met en veille dès la fin du tour** : un
entraînement « 4 h 30 non-stop » n'y est pas possible. Tout est donc construit
pour être **interrompu et repris sans perte** : chaque tranche sauvegarde un
checkpoint complet (modèle + optimiseur + itération + historique).

```bash
SLICE=25 TOTAL=200000 ./scripts/train_forever.sh    # tranches + backup 30 min + sleep 300
```

Sur une machine GPU (ou une instance louée), les mêmes scripts avec
`--batch 24 --crop 256` donnent un vrai modèle.

## 3. Reprendre en 3 commandes

```bash
# 1. deps (la VM repart sans venv)
python3 -m venv /tmp/vfi/.venv && . /tmp/vfi/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. données (7 clips 720p, ~4 min, frames consécutives garanties)
python scripts/fetch_derf.py --frames 300 --out /tmp/vfi/derf

# 3. entraînement (reprend depuis checkpoints/ckpt_last.pt)
export PYTHONPATH=$PWD
python -m src.train --derf-dir /tmp/vfi/derf --val-clips-names "Johnny,shields" \
    --iters 20000 --max-minutes 25 --batch 8 --crop 192 --threads 6 \
    --resume checkpoints/ckpt_last.pt
```

## 4. Feuille de route, par rentabilité décroissante

1. **Plus d'itérations** (facteur limitant n°1) : 3 549 → 50 000+, crop 256, GPU.
   Gain attendu de l'ordre de +2 à +4 dB sur les clips difficiles (`parkrun`, `shields`).
2. **Vraies données Vimeo-90K** : archive officielle (403 depuis la VM, à
   télécharger depuis un navigateur) puis `--vimeo-root/--vimeo-list`.
3. **Architecture** : 4ᵉ niveau de pyramide, blocs à contexte global type
   RIFE-IFNet ; garder le contrat QNN-friendly (pas d'op exotique).
4. **Pertes** : LPIPS/Census (zones occultées), puis QAT int8 pour le HTP.
5. **Mesures réelles** : `qai-hub profile` sur Galaxy S23/S24 → remplir
   `docs/RESULTS.md` (latence, backend par couche, fallbacks CPU).
6. **Android** : module Kotlin `MediaCodec` → interpolation → encodage pour un
   vrai 30→60/120 fps.

## 5. Poids et artefacts

Dans `checkpoints/` (workspace de ce chat, persistant) :
`ckpt_best.pt` (39,41 dB), `ckpt_last.pt` (reprise), `mobileifnet.onnx` +
`mobileifnet.onnx.data`, `mobileifnet_fp16.onnx`, `mobileifnet_int8.onnx`.

Pour les pousser sur GitHub depuis une machine où tu as un token :

```bash
git remote add origin https://github.com/Underruness/dev.git
git add -A && ./scripts/backup.sh "add trained weights"
```

## 6. Chiffres à ne jamais réinventer

* 137 087 params · 0,55 Mo de poids fp32 · entrée 2×3×256×448.
* CPU VM : ~0,45 s/it (batch 8, crop 192²) → ~2,2 it/s.
* Eval : modèle 39,41 dB / blend 33,51 dB / repeat 30,38 dB (596 triplets inédits).
* Miroirs Vimeo-90K HF **mal alignés** (blend 19-25 dB) : ne pas les réutiliser
  sans revérifier le PSNR du mélange linéaire (~30-40 dB attendu).
