# Datasets : ce qui marche, ce qui ne marche pas (mesuré)

## 1. Vimeo-90K officiel : inaccessible depuis cette VM

```bash
curl -I https://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip   # -> 403 Forbidden
```

Idem `vimeo_septuplet.zip`, y compris avec un User-Agent navigateur et une requête
`Range`. L'archive complète ferait de toute façon ~80 Go (la VM a 55 Go libres).

## 2. Miroirs HuggingFace : présents, mais **mal alignés**

| Dataset | Format | Constat mesuré |
|---|---|---|
| `danjacobellis/vimeo90k_triplet` | 71 shards parquet de ~264 Mo, 1 image/ligne + `label` | **frames non consécutives** : PSNR(milieu, blend) ≈ **19 dB**, PSNR(im1, im2) ≈ **17 dB** (attendu : 30-40 dB). Le montage `logs/samples/triplet0.png` montre un travelling de ~80 px entre deux frames « adjacentes ». |
| `bijinc/vimeo-90k-mini` | 14 shards, colonnes `im1/im2/im3` + `sequence_id` | mieux formé mais **mélange linéaire à 25,7 dB** : les frames restent espacées (plusieurs frames d'écart). Inutilisable pour apprendre l'interpolation *fine*. |

Entraîner sur ces données apprend au modèle à inventer un mouvement de 80 px
entre deux images : le résultat est flou et ne correspond à aucun benchmark.

## 3. Ce qui est utilisé ici : Xiph/DERF 720p (frames consécutives garanties)

`scripts/fetch_derf.py` télécharge 7 clips de référence, décode le flux YUV4MPEG2
en numpy (pas besoin de ffmpeg), redimensionne en 448×256 comme Vimeo-90K et
stocke des tableaux `(300, 256, 448, 3) uint8` :

```bash
python scripts/fetch_derf.py --frames 300 --out /tmp/vfi/derf
```

Contrôle de cohérence (mélange linéaire de frames t-1 | t | t+1) :

```
720p50_mobcal_ter        30.7 dB
720p50_parkrun_ter       21.1 dB   <- gros mouvement / flicker
720p50_shields_ter       23.1 dB
720p5994_stockholm_ter   25.7 dB
FourPeople_1280x720_60   38.0 dB
Johnny_1280x720_60       44.0 dB
KristenAndSara_1280x720_60 41.4 dB
```

Ces valeurs sont dans la plage attendue pour des frames réellement adjacentes,
et la difficulté varie : c'est utilisable pour entraîner *et* pour valider
(clips `Johnny` + `shields` tenus à l'écart dans tous les chiffres du dépôt).

## 4. Passer à Vimeo-90K complet (recommandé pour un vrai modèle)

Sur une machine avec GPU et ~100 Go de disque, récupère l'archive officielle
(ou la version septuplet), puis :

```bash
unzip vimeo_triplet.zip -d /data/vimeo90k
python -m src.train --vimeo-root /data/vimeo90k \
    --vimeo-list /data/vimeo90k/tri_trainlist.txt \
    --iters 300000 --batch 24 --crop 256
```

`src/data.py` gère ce layout (`sequences/<clip>/<seq>/im1.png|im2.png|im3.png`) ;
en mode `--vimeo-root` la validation se fait sur tout le jeu fourni, donc pointe
`--vimeo-list` vers `tri_testlist.txt` pour une évaluation honnête (le script
actuel utilise la même liste pour les deux : à ajuster en une ligne si besoin).

Le loader parquet (`--data '*.parquet'`) reste disponible pour les miroirs HF :
il est correct techniquement (triplets reconstruits par label), seules les
données de ces miroirs sont mauvaises.
