# Résultats mesurés (checkpoint `ckpt_best.pt`)

Tous les chiffres ci-dessous viennent de commandes réellement exécutées sur la VM
de ce chat (8 vCPU, **aucun GPU**), pas d'estimations. Reproductibles avec les
commandes de `README.md`.

## Environnement d'entraînement

| Élément | Valeur |
|---|---|
| Machine | Freestyle VM, 8 vCPU x86_64, 16 Go RAM, **0 GPU** |
| Framework | PyTorch 2.15.0+cpu |
| Dataset | 7 clips Xiph/DERF 720p50/60 décodés en `(300, 256, 448, 3) uint8` = **2 086 triplets** consécutifs |
| Split | 1 490 triplets d'entraînement (5 clips), 596 de validation (clips `Johnny` + `720p50_shields` **jamais vus**) |
| Itérations | 3 549 (≈ 26 min de CPU au total, ~0,45 s/it, batch 8, crop 192²) |
| Modèle | `MobileIFNet`, 137 087 paramètres |

## Qualité (596 triplets de validation, clips tenus à l'écart)

| Méthode | PSNR (dB) | SSIM |
|---|---|---|
| Recopier la frame précédente | 30.38 | — |
| Mélange linéaire `0.5·(f0+f1)` | 33.51 | 0.8169 |
| **MobileIFNet (iter 2800)** | **39.41** | **0.9697** |

→ **+5,9 dB de PSNR** et **+0,15 de SSIM** par rapport au mélange linéaire, sur des
clips jamais vus à l'entraînement, avec 137 k paramètres (0,55 Mo de poids fp32).

> Ces clips (vidéoconférence 60 fps + travelling) sont *faciles* pour un modèle à
> flot : le mélange linéaire y est mauvais (33,5 dB) parce que le mouvement est
> important. Sur Vimeo-90K officiel, la référence « linéaire » est ~31 dB et les
> modèles SOTA atteignent ~35-36 dB : nos 39,4 dB **ne sont pas comparables** à un
> score de benchmark public. Voir `docs/DATASET.md` pour la raison (les miroirs
> Vimeo-90K testés sont mal alignés).

## Export / quantification (mesures CPU locales, indicatives)

| Artefact | Taille | Latence ORT-CPU (256×448) |
|---|---|---|
| `mobileifnet.onnx` (fp32 + `.onnx.data`) | 15 Ko + 2,3 Mo | 40,0 ms |
| `mobileifnet_fp16.onnx` | 1,2 Mo | 132,1 ms (émulé, non optimisé) |
| `mobileifnet_int8.onnx` (QDQ) | 2,0 Mo | 35,3 ms |

Différence numérique PyTorch ↔ ONNX vérifiée à l'export : `max abs diff = 1,5e-05`.

⚠️ **Aucune mesure sur téléphone n'a été faite** : il n'y a pas d'appareil
connecté à cette VM. Les latences Snapdragon doivent être obtenues via
`qai-hub profile` (voir `android/README.md`) — ne pas extrapoler les chiffres CPU
ci-dessus à un HTP.

## Courbe de validation (extrait de `logs/val.csv`)

```
iter   psnr_modele  psnr_blend
 800      29.99        21.99     <- échantillon partiel (12 batchs)
1200      34.44        21.99
1600      34.78        21.99
2000      32.83        21.99
2400      34.25        21.99
2800      35.04        21.99
3200      34.55        21.99
```

Les oscillations viennent d'un jeu de validation partiel (12 batchs) et d'un
learning rate encore élevé (3e-4). L'évaluation complète (596 triplets) donne
39,41 dB avec le meilleur checkpoint.

## Ce que ça vaut / ce que ça ne vaut pas

* ✅ Le pipeline complet fonctionne de bout en bout : données → entraînement
  reprenable → évaluation vs baselines → ONNX → int8/fp16 → doc QNN.
* ✅ Le modèle bat très nettement le mélange linéaire sur des clips inédits.
* ❌ Ce n'est **pas** un modèle « meilleur du marché » : 3 549 itérations de CPU
  sur 1 490 triplets, c'est ~0,1 % du budget d'un RIFE/EMA-VFI entraîné sur
  Vimeo-90K (300 k itérations, 8-24 Go de VRAM).
* ❌ Rien n'a été exécuté sur un téléphone : pas de latence réelle, pas de
  validation QNN/GPU delegate.
