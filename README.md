# MobileIFNet — interpolation de frames pour téléphone (QNN / GPU)

Modèle d'interpolation de frames vidéo (VFI) **compact et exportable** vers le
NPU Hexagon via **QNN** (Qualcomm AI Hub), ou vers le GPU mobile (ORT / TFLite
GPU delegate). Entraîné ici, sur cette VM, sur des triplets de frames
réellement consécutives.

**Résultat mesuré** : `39,41 dB` PSNR / `0,970` SSIM sur 596 triplets de
validation inédits, contre `33,51 dB` pour le mélange linéaire → **+5,9 dB**
avec **137 087 paramètres** (0,55 Mo de poids fp32). Détails : [`docs/RESULTS.md`](docs/RESULTS.md).

> ### À lire avant tout (honnêteté sur le périmètre)
> * Cette VM **n'a pas de GPU** (8 vCPU, 16 Go RAM, pas de `/dev/nvidia*`).
>   Les 3 549 itérations faites ici, c'est ~26 min de CPU : un **démarrage de
>   pipeline**, pas un modèle « meilleur du marché ». RIFE/EMA-VFI = 10-100 M
>   paramètres et des jours de GPU.
> * La VM **se met en veille dès que le tour se termine** : aucun process ne
>   peut y tourner 4 h 30. `scripts/train_forever.sh` est conçu pour ça —
>   tranches bornées + checkpoint + backup + `sleep 300`, donc **tuable et
>   reprenable** à tout instant, sans perdre une itération.
> * Le Vimeo-90K officiel renvoie **403** et les miroirs HF testés ont des
>   frames **non adjacentes** (mélange linéaire à 19-25 dB au lieu de ~31 dB).
>   J'ai donc entraîné sur des clips Xiph/DERF 720p dont la consécutivité est
>   garantie. Tout est détaillé et mesuré dans [`docs/DATASET.md`](docs/DATASET.md).
> * **Aucun test sur téléphone** n'a été fait (pas d'appareil connecté) : les
>   latences HTP/GPU restent à mesurer, procédure exacte dans
>   [`android/README.md`](android/README.md).

## 1. Le modèle

`src/model.py` — IFNet-lite coarse-to-fine, 3 niveaux, flot bidirectionnel + masque :

```
frame0, frame1  (3+3 canaux)
      │
   enc0 (24ch, /1) ──► enc1 (48ch, /2) ──► enc2 (64ch, /4)
                                            │
                          head2: flow(4) + mask(1) @ /4
                                            │ ×2 + résidu
                                fuse1 + head1 @ /2
                                            │ ×2 + résidu
                                fuse0 + head0 @ /1
                                            │
              warp(frame0, flow0↔) + warp(frame1, flow1↔), mélange par mask
                                            │
                                    frame_mid (3 canaux)
```

* **Que des ops QNN-friendly** : Conv2d (stride 1/2), LeakyReLU, Concat, Resize
  bilinéaire, Sigmoid, Mul/Add, un seul `GridSample`. Pas de déformable conv,
  pas de matmul exotique, pas de LayerNorm → pas de fallback CPU sur le HTP.
* Deep supervision à 3 échelles (poids 0,25 / 0,5 / 1,0), perte de Charbonnier.
* 137 087 paramètres ; artefacts ONNX : 2,3 Mo fp32 (+15 Ko de graphe),
  1,2 Mo fp16 (à donner à AI Hub/QNN), 2,0 Mo int8 QDQ.
* Entrée de déploiement : **256×448**, NCHW fp32, 2 entrées (`frame0`,
  `frame1`) → 1 sortie (`frame_mid`).

## 2. Les données

```bash
# clips Xiph/DERF 720p -> (300, 256, 448, 3) uint8, frames consécutives garanties
python scripts/fetch_derf.py --frames 300 --out /tmp/vfi/derf   # ~100 Mo/clip
```

* `src/data.py` → `NpyClips` : triplets `(t, t+1, t+2)` **dans un même clip**,
  clips de validation tenus à l'écart (`--val-clips-names`).
* `src/data.py` → `ParquetTriplets` / `FolderTriplets` : loader Vimeo-90K
  (shards HF parquet, ou layout officiel `vimeo_triplet/` via
  `--vimeo-root/--vimeo-list`) pour quand tu auras les vraies données.
* Augmentation : crops aléatoires, flip horizontal, inversion temporelle.

Pourquoi pas Vimeo-90K direct : voir [`docs/DATASET.md`](docs/DATASET.md).

## 3. Entraîner

```bash
python3 -m venv /tmp/vfi/.venv && . /tmp/vfi/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# une tranche (crash-safe, reprend automatiquement)
python -m src.train --derf-dir /tmp/vfi/derf \
    --val-clips-names "Johnny,shields" \
    --iters 20000 --max-minutes 25 --batch 8 --crop 192 --threads 6 \
    --resume checkpoints/ckpt_last.pt
```

Sur GPU (le seul moyen d'aller vers un vrai modèle) :

```bash
python -m src.train --derf-dir /data/derf --iters 200000 \
    --batch 24 --crop 256 --val-every 2000 --ckpt-every 2000
```

Boucle continue + backups 30 min (le « sleep 5 min » demandé) :

```bash
nohup ./scripts/train_forever.sh >> logs/supervisor.log 2>&1 &
```

Sorties : `checkpoints/ckpt_last.pt` (reprise), `checkpoints/ckpt_best.pt`,
`logs/train.csv`, `logs/val.csv`, `logs/best.json`.

## 4. Évaluer

```bash
python -m src.eval --ckpt checkpoints/ckpt_best.pt --derf-dir /tmp/vfi/derf \
    --val-clips-names "Johnny,shields" --save-samples 4
```

Écrit `logs/eval.json` et des comparatifs `logs/samples/cmp_XX.png`
(**gt | mélange linéaire | modèle**).

## 5. Exporter / quantifier (vers le téléphone)

```bash
python -m src.export_onnx --ckpt checkpoints/ckpt_best.pt --out checkpoints/mobileifnet.onnx
python -m src.quantize --onnx checkpoints/mobileifnet.onnx --derf-dir /tmp/vfi/derf
```

→ `mobileifnet.onnx` (+ `.onnx.data`), `mobileifnet_fp16.onnx` (pour AI Hub /
QNN), `mobileifnet_int8.onnx` (QDQ). Vérification numérique PyTorch↔ONNX faite à
l'export (`max abs diff 1,5e-05`).

Déploiement : **[`android/README.md`](android/README.md)** — AI Hub → contexte QNN
fp16 Hexagon, onnxruntime + QNN EP dans l'app, ou TFLite GPU delegate.

## 6. Limites assumées

| Attendu | Réalité |
|---|---|
| Modèle « meilleur du marché » | 137 k params, 3 549 itérations CPU ; SOTA = GPU indispensable |
| Tourner 4 h 30 sans arrêt | impossible dans cette VM (veille) ; supervisor reprenable à la place |
| Vimeo-90K officiel | 403 + miroirs mal alignés → dataset DERF, voir `docs/DATASET.md` |
| Latence/batterie téléphone | **non mesurées** (aucun appareil connecté) |
| Qualité du checkpoint | 39,41 dB vs 33,51 dB (blend) sur clips inédits — voir `docs/RESULTS.md` |

## 7. Structure

```
src/model.py         réseau (137 k params)
src/data.py          NpyClips (frames consécutives) + loaders Vimeo-90K
src/train.py         boucle reprenable, deep supervision, PSNR/SSIM
src/eval.py          évaluation + baselines + images gt|blend|modèle
src/export_onnx.py   export ONNX + vérif numérique
src/quantize.py      int8 QDQ / fp16 + bench ORT
scripts/fetch_derf.py      clips 720p -> .npy (frames consécutives)
scripts/fetch_vimeo90k.py  shards parquet Vimeo-90K (HF)
scripts/train_forever.sh   supervisor (tranches + backups + sleep 300)
scripts/backup.sh          git commit/push (garde-fou > 20 Mo)
android/README.md          QNN (AI Hub), ORT QNN EP, TFLite GPU
docs/RESULTS.md            chiffres mesurés
docs/DATASET.md            pourquoi DERF et pas les miroirs Vimeo-90K
HANDOFF.md                 reprise dans une nouvelle discussion
```
