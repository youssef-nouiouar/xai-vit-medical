# Evaluation XAI — Phase 1
## Classification histologique CRC · ResNet-50 vs. ViT-Base / DeiT-Base / DINOv2 / Swin-Base

> **Auteur :** Youssef Nouiouar
> ### 🧬 Dataset — Histological Colorectal Cancer (CRC) |**➡️ [Disponible :Zenodo](https://zenodo.org/records/1214456)**  
> **Dataset test :** CRC-VAL-HE-7K — 7 180 images, 9 classes (sous-ensemble : 50 images/classe = 450 images)  
> **Environnement :** Kaggle GPU T4 / P100  
> **Phase :** Phase 1 uniquement — histologie 2D CRC

---

## Table des matières

1. [Performance de classification](#1-performance-de-classification)
2. [Fidélité des explications — Insertion / Deletion AUC](#2-fidélité-des-explications--insertion--deletion-auc)
   - 2.1 [ResNet-50](#21-resnet-50)
   - 2.2 [ViT-Base/16](#22-vit-base16)
   - 2.3 [DeiT-Base](#23-deit-base)
   - 2.4 [DINOv2-ViT-B/14](#24-dinov2-vit-b14)
   - 2.5 [Swin-Base](#25-swin-base)
   - 2.6 [Synthèse globale toutes combinaisons](#26-synthèse-globale-toutes-combinaisons)
3. [Comparaison CNN vs. ViT — GradCAM & Integrated Gradients](#3-comparaison-cnn-vs-vit--gradcam--integrated-gradients)
4. [Apport de Generic Attention sur les ViTs](#4-apport-de-generic-attention-sur-les-vits)
5. [Inspection visuelle — montages par classe](#5-inspection-visuelle--montages-par-classe)
6. [Remarques et conclusions générales](#6-remarques-et-conclusions-générales)
7. [Méthodes mécanistes — SAE (Sparse Autoencoders)](#7-méthodes-mécanistes--sae-sparse-autoencoders)
   - 7.1 [Pipeline et notebooks](#71-pipeline-et-notebooks)
   - 7.2 [Choix de configuration](#72-choix-de-configuration)
   - 7.3 [Résultats SAE — DeiT-Base](#73-résultats-sae--deit-base)
   - 7.4 [Résultats SAE — DINOv2-ViT-B/14](#74-résultats-sae--dinov2-vit-b14)
   - 7.5 [Patches les plus activants par classe](#75-patches-les-plus-activants-par-classe)
   - 7.6 [Espace des features SAE — t-SNE](#76-espace-des-features-sae--t-sne)
   - 7.7 [Analyse d'ablation causale](#77-analyse-dablation-causale)
   - 7.8 [Discussion et limites](#78-discussion-et-limites)

---

## 1. Performance de classification

> Mesure de l'accuracy top-1 de chaque modèle sur le sous-ensemble de test (450 images).  
> L'accuracy conditionne la validité des explications XAI : une carte saliency sur une prédiction erronée n'a pas de valeur explicative directe.

| Modèle | Architecture | Nb paramètres | Accuracy top-1 (%) | Remarque |
|---|---|---|---|---|
| ResNet-50 | CNN | 25M | 0.91 | Baseline CNN |
| ViT-Base/16 | ViT |  86M | 0.926 | Ancrage SOTA |
| DeiT-Base | ViT data-efficient | 85M | 0.944 | |
| DINOv2-ViT-B/14 | ViT auto-supervisé | 86M | 0.947 | Sonde linéaire |
| Swin-Base | ViT hiérarchique | 86M | 0.957 | |

> **Remarque :** Tous les modèles atteignent une précision élevée sur le jeu de données choisi pour l’évaluation. Cela montre que la comparaison des performances n’est pas biaisée par la qualité du jeu de données, et que les différences observées reflètent bien les capacités propres des architectures.

---

## 2. Fidélité des explications — Insertion / Deletion AUC

> **Rappel des métriques :**
> - **Insertion AUC** : les pixels les plus saillants sont révélés progressivement sur fond noir. Une valeur élevée indique que la carte localise bien les régions décisives (la confiance remonte vite).
> - **Deletion AUC** : les pixels les plus saillants sont supprimés progressivement. Une valeur faible est souhaitable (la confiance chute vite si les zones importantes sont retirées).
> - **Faithfulness = Insertion AUC − Deletion AUC** : métrique principale de fidélité causale. Plus elle est élevée, plus l'explication est causalement alignée avec la décision du modèle.
> - Baseline : `black` (pixels noirs) pour insertion et deletion.
> - Nombre de pas : 50 (`FAITH_N_STEPS = 50`).

### 2.1 ResNet-50

| Méthode XAI | Insertion AUC | Deletion AUC | Faithfulness |
|---|---|---|---|
| GradCAM | 0.757 | 0.559 | 0.197 |
| Integrated Gradients | 0.381 | 0.286 | 0.094 |
| LRP | 0.365 | 0.310 | 0.054 |
| Attention Rollout | — | — | — |
| Generic Attention | — | — | — |

> **Résumé :** GradCAM est la méthode la plus fidèle sur CNN. LRP et IG restent en retrait.


---

### 2.2 ViT-Base/16

| Méthode XAI | Insertion AUC | Deletion AUC | Faithfulness |
|---|---|---|---|
| GradCAM | 0.612 | 0.374 | 0.237 |
| Integrated Gradients | 0.507 | 0.332 | 0.175 |
| Attention Rollout | 0.565 | 0.515 | 0.05 |
| Generic Attention ⭐ | 0.661 | 0.385 | 0.275 |
| LRP | 0.471 | 0.377 | 0.094 |


---

### 2.3 DeiT-Base

| Méthode XAI | Insertion AUC | Deletion AUC | Faithfulness |
|---|---|---|---|
| GradCAM | 0.892 | 0.698 | 0.193 |
| Integrated Gradients | 0.615 | 0.522 | 0.093 |
| Attention Rollout | 0.839 | 0.837 | 0.002 |
| Generic Attention ⭐ | 0.911 | 0.660 | 0.251 |
| LRP | 0.586 | 0.570 | 0.015 |

> **Note :** 2 tokens spéciaux ([CLS] + [DIST]) pris en compte pour Rollout et Generic Attention.
---

### 2.4 DINOv2-ViT-B/14

| Méthode XAI | Insertion AUC | Deletion AUC | Faithfulness |
|---|---|---|---|
| GradCAM | 0.824 | 0.619 | 0.204 |
| Integrated Gradients | 0.470 | 0.368 | 0.102 |
| Attention Rollout | 0.803 | 0.752 | 0.051 |
| Generic Attention ⭐ | 0.846 | 0.585 | 0.261 |
| LRP | 0.452 | 0.395 | 0.056 |

> **Résumé :** L'auto-supervision DINO produit des représentations structurées, mais la sonde linéaire limite la qualité des gradients pour IG et LRP (faithfulness faible). Generic Attention reste la méthode la plus fiable.


---

### 2.5 Swin-Base

| Méthode XAI | Insertion AUC | Deletion AUC | Faithfulness |
|---|---|---|---|
| GradCAM | 0.909 | 0.902 | 0.006 |
| Integrated Gradients | 0.590 | 0.556 | 0.034 |
| LRP | 0.590 | 0.637 | -0.047 |
| Attention Rollout | — | — | — |
| Generic Attention | — | — | — |

> **Note :** GradCAM est inutilisable sur Swin (résolution 7×7, insertion et deletion quasi identiques). IG domine mais reste faible en absolu. Rollout et Generic Attention sont inapplicables (attention fenêtrée).


---

### 2.6 Synthèse globale toutes combinaisons


| Modèle | GradCAM Faith. | IG Faith. | Rollout Faith. | Generic Att. Faith. ⭐ | LRP Faith. |
|---|---|---|---|---|---|
| ResNet-50 | 0.197 | 0.094 | — | — | 0.054 |
| ViT-Base/16 | 0.237 | 0.175 | 0.05 | 0.275 | 0.094 |
| DeiT-Base | 0.193 | 0.093 | 0.02 | 0.251 | 0.015 |
| DINOv2-ViT-B/14 | 0.204 | 0.102 | 0.051 | 0.261 | 0.056 |
| Swin-Base | 0.006 | 0.034 | — | — | -0.047 |

---
**Total :** 21 combinaisons actives sur 21 possibles.

## 3. Comparaison CNN vs. ViT — GradCAM

> Seule comparaison inter-architectures retenue : GradCAM est la méthode commune la plus performante sur CNN et applicable à tous les modèles.

| Modèle | Famille | Insertion AUC | Deletion AUC | Faithfulness | Résolution carte brute |
|---|---|---|---|---|---|
| ResNet-50 | CNN | 0.757 | 0.559 | 0.197 | 14×14 |
| ViT-Base/16 | ViT | 0.612 | 0.374 | 0.237 | 14×14 |
| DeiT-Base | ViT |0.892 | 0.698 | 0.193 | 14×14 |
| DINOv2-ViT-B/14 | ViT | 0.824 | 0.619 | 0.204 | 16×16 |
| Swin-Base | ViT hiérarchique | 0.909 | 0.902 | 0.006 | 7×7 |

> **Résumé  :** GradCAM est légèrement plus fidèle sur ViT-Base (0.237) que sur ResNet (0.197). Swin est pénalisé par la résolution 7×7 (faithfulness quasi nulle). Le reshape_transform sur les ViTs standard n'introduit pas de biais majeur.

---

## 4. Apport de Generic Attention sur les ViTs
### 4.1 Generic Attention vs. Attention Rollout

| Modèle          | Rollout Faith. | GenAtt Faith. | Δ          |
| --------------- | -------------- | ------------- | ---------- |
| ViT-Base/16     | 0.050          | 0.275         | **+0.225** |
| DeiT-Base       | 0.002          | 0.251         | **+0.249** |
| DINOv2-ViT-B/14 | 0.051          | 0.261         | **+0.210** |


> **Résumé :** Generic Attention améliore systématiquement la fidélité de +0.21 à +0.25. Rollout, avec discard_ratio=0.9, est trop agressif et élimine des régions décisives

### 4.2 Generic Attention vs. GradCAM sur les ViTs

| Modèle          | GradCAM Faith. | GenAtt Faith. | Dominante             |
| --------------- | -------------- | ------------- | --------------------- |
| ViT-Base/16     | 0.237          | 0.275         | **Generic Attention** |
| DeiT-Base       | 0.193          | 0.251         | **Generic Attention** |
| DINOv2-ViT-B/14 | 0.204          | 0.261         | **Generic Attention** |


> **Résumé :** Generic Attention exploite la structure interne d'attention du Transformer ; GradCAM reste un proxy externe via les gradients de feature maps. L'écart se confirme numériquement sur les 3 modèles.

### 4.3 Observations qualitatives — Generic Attention

>**Résumé :**  Les cartes Generic Attention sont plus localisées que Rollout et alignées sur des structures biologiques plausibles (contours de glandes dans ADI, noyaux dans TUM). Elles échouent davantage sur les classes homogènes (BACK, DEB) où l'absence de structure discriminante fragmente la saillance.

<img src="outputs/xai/dinov2/generic_attention/generic_attention_demonstration.png" alt="Montage DINOv2 — GradCAM" width=""/>

---

## 5. Inspection visuelle — montages par classe 
> Les classes CRC sont : ADI · BACK · DEB · LYM · MUC · MUS · NORM · STR · TUM.

### 5.1 ResNet-50
| GradCAM | IG | LRP |
|--------|----|-----|
| <img src="outputs/xai/resnet50/gradcam/class_montage.png" width="250"/> | <img src="outputs/xai/resnet50/integrated_gradients/class_montage.png" width="250"/> | <img src="outputs/xai/resnet50/lrp/class_montage.png" width="250"/> |

---

### 5.2 ViT-Base/16
| GradCAM |      IG          | Rollout  | Generic Attention | LRP |
|--------|-------------------|----------|-------------------|-----|
| <img src="outputs/xai/vit_base/gradcam/vit_base_class_montage.png" width="160"/> | <img src="outputs/xai/vit_base/integrated_gradients/vit_base_class_montage.png" width="160"/> |<img src="outputs/xai/vit_base/attention_rollout/vit_base_class_montage.png" width="160"/> | <img src="outputs/xai/vit_base/generic_attention/vit_base_class_montage.png" width="160"/> | <img src="outputs/xai/vit_base/lrp/vit_base_class_montage.png" width="160"/> |
  
---

### 5.3 DeiT-Base
| GradCAM | IG | Rollout | Generic Attention | LRP |
|--------|----|----------|-------------------|-----|
| <img src="outputs/xai/deit_base/gradcam/deit_base_class_montage.png" width="160"/> | <img src="outputs/xai/deit_base/integrated_gradients/deit_base_class_montage.png" width="160"/> | <img src="outputs/xai/deit_base/attention_rollout/deit_base_class_montage.png" width="160"/> | <img src="outputs/xai/deit_base/generic_attention/deit_base_class_montage.png" width="160"/> | <img src="outputs/xai/deit_base/lrp/deit_base_class_montage.png" width="160"/> |
  
---

### 5.4 DINOv2-ViT-B/14
| GradCAM | IG | Rollout | Generic Attention | LRP |
|--------|----|----------|-------------------|-----|
| <img src="outputs/xai/dinov2/gradcam/dinov2_class_montage.png" width="160"/> | <img src="outputs/xai/dinov2/integrated_gradients/dinov2_class_montage.png" width="160"/> | <img src="outputs/xai/dinov2/attention_rollout/dinov2_class_montage.png" width="160"/> | <img src="outputs/xai/dinov2/generic_attention/dinov2_class_montage.png" width="160"/> | <img src="outputs/xai/dinov2/lrp/dinov2_class_montage.png" width="160"/> |
  
---

### 5.5 Swin-Base
| GradCAM | IG | LRP |
|--------|----|-----|
| <img src="outputs/xai/swin_base/gradcam/swin_base_class_montage.png" width="260"/> |  <img src="outputs/xai/swin_base/integrated_gradients/swin_base_class_montage.png" width="260"/> | <img src="outputs/xai/swin_base/lrp/swin_base_class_montage.png" width="260"/> |

---

## 6. Conclusions générales


### 6.1 Performances de classification

> Les ViTs surpassent ResNet-50. DINOv2 (sonde linéaire, 0.947) est compétitif malgré un entraînement supervisé limité. Swin-Base se distingue légèrement (0.957).

### 6.2 Méthode XAI la plus fidèle par architecture

| Architecture    | Méthode dominante     | Faithfulness |
| --------------- | --------------------- | ------------ |
| ResNet-50       | GradCAM               | 0.197        |
| ViT-Base/16     | **Generic Attention** | 0.275        |
| DeiT-Base       | **Generic Attention** | 0.251        |
| DINOv2-ViT-B/14 | **Generic Attention** | 0.261        |
| Swin-Base       | Integrated Gradients  | 0.034        |

> **Tendance :** Pas de méthode universelle. Generic Attention domine sur tous les ViTs à attention globale. GradCAM reste le choix par défaut sur CNN. Swin-Base reste problématique pour l'interprétabilité.

### 6.3 CNN vs. ViT — interprétabilité

> Les ViTs sont plus interprétables que ResNet-50 à condition d'utiliser Generic Attention. Les cartes ViT sont visuellement plus structurées mais plus diffuses (granularité patch). L'attention apporte une valeur ajoutée claire par rapport aux gradients seuls.

### 6.4 Apport de Generic Attention

> Gain massif et systématique vs Rollout (+0.21 à +0.25). Surpasse GradCAM sur tous les ViTs testés. Justifie pleinement son usage comme méthode principale dans le papier MIDL 2026.


### 6.5 Limites et biais 

| Biais                             | Impact                                            |
| --------------------------------- | ------------------------------------------------- |
| Sous-ensemble 450 images          | Tendance fiable, mais généralisation limitée      |
| Baseline noire insertion/deletion | Inadaptée à l'histologie (fonds colorés naturels) |
| Sonde linéaire DINOv2             | Gradients moins informatifs pour IG/LRP           |
| `discard_ratio=0.9` (Rollout)     | Trop agressif, élimine des zones pertinentes      |
| GradCAM sur Swin (7×7)            | Résolution trop faible pour l'histologie          |


### 6.6 Perspectives — étapes suivantes

> - Sanity checks (model randomization + label randomization) — obligatoires avant publication
> - SAE sur DeiT-Base et DINOv2 — **réalisé, voir [Section 7](#7-méthodes-mécanistes--sae-sparse-autoencoders)** ; résultats à améliorer (faible fréquence d'activation)
> - Activation Patching 
> - Analyse approfondie des classes difficiles : MUC, DEB

---

## 7. Méthodes mécanistes — SAE (Sparse Autoencoders)

> **Objectif :** Décomposer les représentations internes des ViTs en features monosémantiques interprétables, puis localiser spatialement leur activation sur les tissus histologiques CRC.

---

### 7.1 Pipeline et notebooks

| Étape | Notebook | Description |
|-------|----------|-------------|
| 1 — Extraction | `06_collect_activations.ipynb` | Extraction des activations patch à la couche cible ; sauvegarde au format HDF5 (`*_layer9_acts.h5`) |
| 2 — Entraînement | `06_training_sae.ipynb` | Entraînement du SAE TopK sur les activations collectées |
| 3 — Interprétation | `06_interpretat_featers.ipynb` | Statistiques d'activation, fréquence par classe, cartes spatiales, ablation causale, t-SNE |

**Résultats stockés dans :** `outputs/sae/`

---

### 7.2 Choix de configuration

#### Couche primaire : Layer 9

La couche 9 a été retenue pour deux raisons complémentaires :

- Dans un modèle finement ajusté sur une tâche de classification histologique, les couches 9 et 10 encodent les distinctions propres au tissu colorectal (tumeur vs. tissu normal vs. tissu sain, etc.). Ce sont les couches « conceptuelles » du réseau.
- La couche 11 est trop proche de la tête de classification : les features y deviennent des proxies de logits de classe plutôt que des représentations sémantiques indépendantes. Un SAE entraîné sur cette couche apprendrait des directions liées à la prédiction finale, pas à des concepts tissulaires.

#### `token_type = patch` (et non CLS)

| | CLS token | Patch tokens |
|---|---|---|
| Dimensionnalité | 1 vecteur × 768 dims | 196 vecteurs (DeiT, patch 16×16) / 256 vecteurs (DINOv2, patch 14×14) |
| Information spatiale | Aucune — résumé global de l'image | Oui — chaque vecteur correspond à une région de 16×16 px ou 14×14 px |
| Visualisation SAE | Non localisable | Carte 2D de l'activation de chaque feature sur le tissu |
| Précédent | SAE-Rad (Li et al.) | Approche adoptée ici |

Le choix des patch tokens est central pour l'histologie : l'objectif est de montrer « cette feature SAE s'active sur les lumières glandulaires » (ADI) ou « sur les noyaux tumoraux » (TUM), pas seulement « sur les images cancéreuses ». La localisation spatiale est indispensable pour que les features soient cliniquement interprétables. Le CLS token pourra faire l'objet d'une expérience complémentaire ultérieure.

#### Hyperparamètres SAE — v2 (DeiT-Base)

> **Note :** Les paramètres v1 (EXPANSION=16, k=32, 5 époques) produisaient des features avec une fréquence d'activation quasi nulle. La v2 corrige ces trois points. Voir section 7.8 pour le diagnostic complet.

| Paramètre | v1 ~~(abandonné)~~ | **v2 (actuel)** | Raison du changement |
|-----------|-------------------|-----------------|----------------------|
| Architecture | TopK SAE | **TopK SAE** | — |
| `d_input` | 768 | **768** | Dimension cachée ViT-B |
| Expansion | ~~16×~~ | **8×** | Dictionnaire trop large pour 25 k images ; réduit la compétition entre features |
| `d_sae` | ~~12 288~~ | **6 144** | 768 × 8 |
| `k` (TopK) | ~~32~~ | **64** | Taux de tirage 0.26 % → 1.04 % par token ; double directement la fréquence |
| `encode()` ReLU | ~~clamp après TopK~~ | **ReLU avant TopK** | v1 gaspillait des slots sur des valeurs négatives, L0 effectif < k |
| Init W_enc | ~~Kaiming indépendant~~ | **W_enc = W_dec.T** | Directions conjuguées dès l'initialisation |
| Taux d'apprentissage | ~~3×10⁻⁴~~ | **2×10⁻⁴** | Légèrement réduit pour k plus grand |
| Époques | ~~5~~ | **15** | Plus de passes nécessaires pour la spécialisation |
| `warmup_steps` | ~~1 000~~ | **2 000** | Mis à l'échelle des ~16 k pas totaux |
| `aux_k` | ~~64~~ | **128** | ≈ 2×k |
| `dead_window` | ~~200~~ | **500** | Évite de réactiver des features en phase de chauffage |
| Batch size | 4 096 tokens | **4 096 tokens** | — |
| `aux_coef` | 0.25 | **0.25** | — |

---

### 7.3 Résultats SAE — DeiT-Base

> **Fichiers :** `outputs/sae/deit_base/`

#### Qualité de reconstruction — v1 vs v2

| Métrique | v1 (EXPANSION=16, k=32, 5 ep.) | **v2 (EXPANSION=8, k=64, 15 ep.)** | Cible |
|----------|-------------------------------|-------------------------------------|-------|
| R² (variance expliquée) | 0.921 | **0.936** | > 0.85 ✓ |
| Similarité cosinus | 0.952 | **[à compléter v2]** | > 0.90 |
| Features mortes | 28.3 % | **0.0 %** | 10–30 % |
| val MSE final | — | **0.0637** | décroissant ✓ |
| Tokens traités | 4 866 484 | 4 866 484 | — |

> **Note sur dead=0.0 % en v2 :** Avec k=64 sur 6 144 features et ~487 k tokens de validation, chaque feature reçoit statistiquement ~5 000 tirages par passe d'évaluation — il est mathématiquement impossible qu'une feature reste à zéro. Ce n'est pas un problème : cela indique que le dictionnaire est entièrement utilisé. En v1 (k=32/12 288 = 0.26 % par token), de nombreuses features étaient statistiquement affamées.

| Époque | val MSE | R² |
|--------|---------|:--:|
| 1  | 0.1739 | 0.826 |
| 5  | 0.0712 | 0.929 |
| 10 | 0.0646 | 0.935 |
| **15** | **0.0637** | **0.936** |

![Courbes d'entraînement DeiT-Base v2](outputs/sae/deit_base/sae_deit_base_patch16_training.png)

#### Top feature par classe — fréquence et sélectivité

> Les colonnes `freq_tok` et `freq_img` ci-dessous sont celles de v1. Relancer `06_interpretat_featers.ipynb` avec le checkpoint v2 pour obtenir les valeurs v2 (freq_img attendu : nettement plus élevé grâce à k=64).

| Classe | Feature (v1) | Sélectivité | freq_tok v1 | freq_img v2 |
|--------|-------------|:-----------:|:-----------:|:-----------:|
| ADI  | #1198  | 1.000 | 0.01 % | [à compléter] |
| BACK | #4498  | 1.000 | 0.02 % | [à compléter] |
| DEB  | #566   | 0.999 | < 0.01 % | [à compléter] |
| LYM  | #10866 | 1.000 | 0.31 % | [à compléter] |
| MUC  | #2835  | 1.000 | < 0.01 % | [à compléter] |
| MUS  | #5147  | 1.000 | 0.01 % | [à compléter] |
| NORM | #1031  | 1.000 | 0.69 % | [à compléter] |
| STR  | #9967  | 1.000 | 0.06 % | [à compléter] |
| TUM  | #11685 | 1.000 | 0.03 % | [à compléter] |

---

### 7.4 Résultats SAE — DINOv2-ViT-B/14

> **Fichiers :** `outputs/sae/dinov2/`

#### Qualité de reconstruction (v1)

| Métrique | Valeur | Cible |
|----------|--------|-------|
| R² (variance expliquée) | **0.940** | > 0.85 ✓ |
| Similarité cosinus | **0.957** | > 0.90 ✓ |
| Features mortes | **30.3 %** | 10–30 % ✓ |
| Tokens traités | 10 199 296 | (39 841 images × 256 patches) |

![Courbes d'entraînement DINOv2](outputs/sae/dinov2/sae_dinov2_base_patch16_training.png)

#### Top feature par classe — fréquence et sélectivité (v1)

| Classe | Feature | Sélectivité | Fréquence d'activation (token-level) |
|--------|---------|:-----------:|:------------------------------------:|
| ADI  | #247   | 1.000 | 0.004 % |
| BACK | #8651  | 1.000 | 0.004 % |
| DEB  | #10972 | 0.999 | < 0.001 % |
| LYM  | #4505  | 0.999 | 13.69 % |
| MUC  | #2650  | 0.999 | 0.001 % |
| MUS  | #11772 | 0.999 | 0.001 % |
| NORM | #7452  | 0.999 | < 0.001 % |
| STR  | #4716  | 1.000 | 0.003 % |
| TUM  | #9350  | 0.999 | < 0.001 % |

> **Observation :** Fréquences token-level très faibles sauf pour LYM #4505 (13.69 %), qui représente une exception notable — DINOv2 a apparemment condensé la représentation des lymphocytes dans une feature dédiée forte. Les autres classes restent en dessous de 0.01 %, empêchant une interprétation robuste. Ces valeurs seront recalculées en image-level (freq_img) lors de la relance v2.

---

### 7.5 Patches les plus activants par classe

> Pour chaque feature de tête par classe, les 9 patches qui l'activent le plus fortement sont affichés (100 images par classe). Si les patches d'une feature sont visuellement cohérents (ex. : toutes des lumières glandulaires, tous des lymphocytes…), la feature est monosémantique.

| Classe | DeiT-Base | DINOv2-ViT-B/14 |
|--------|-----------|-----------------|
| ADI  | <img src="outputs/sae/deit_base/interpretations/topk_feat1198.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat247.png" width="220"/> |
| BACK | <img src="outputs/sae/deit_base/interpretations/topk_feat4498.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat8651.png" width="220"/> |
| DEB  | <img src="outputs/sae/deit_base/interpretations/topk_feat566.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat10972.png" width="220"/> |
| LYM  | <img src="outputs/sae/deit_base/interpretations/topk_feat10866.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat4505.png" width="220"/> |
| MUC  | <img src="outputs/sae/deit_base/interpretations/topk_feat2835.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat2650.png" width="220"/> |
| MUS  | <img src="outputs/sae/deit_base/interpretations/topk_feat5147.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat11772.png" width="220"/> |
| NORM | <img src="outputs/sae/deit_base/interpretations/topk_feat1031.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat7452.png" width="220"/> |
| STR  | <img src="outputs/sae/deit_base/interpretations/topk_feat9967.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat4716.png" width="220"/> |
| TUM  | <img src="outputs/sae/deit_base/interpretations/topk_feat11685.png" width="220"/> | <img src="outputs/sae/dinov2/interpretations/topk_feat9350.png" width="220"/> |

> **Résumé :** LYM et NORM (DeiT uniquement) présentent des features genuinement monosémantiques — le SAE a appris des features dédiées avec une activation forte et cohérente. Pour les autres classes (ADI, DEB, MUC…), les patches sont moins cohérents, confirmant la faible fréquence d'activation.

---

### 7.6 Espace des features SAE — t-SNE

> Projection t-SNE des activations SAE moyennées par image, colorées par classe. Des clusters séparés indiquent que le SAE encode une structure discriminante entre classes.

| DeiT-Base | DINOv2-ViT-B/14 |
|-----------|-----------------|
| <img src="outputs/sae/deit_base/interpretations/tsne_sae_features.png" width="350"/> | <img src="outputs/sae/dinov2/interpretations/tsne_sae_features.png" width="350"/> |

> **Observation :** Pour les deux modèles, des clusters séparés par classe sont visibles dans l'espace SAE, ce qui confirme que le SAE capture une structure discriminante malgré les faibles fréquences d'activation par token individuel.

---

### 7.7 Analyse d'ablation causale

> **Principe :** Pour valider qu'une feature SAE est causalement impliquée dans une prédiction, on met à zéro son activation dans le code SAE, on décode le résidu modifié, puis on observe si la prédiction d'une sonde linéaire change. Une sonde linéaire entraînée sur les activations reconstruites sert de classificateur de référence (accuracy = 100 %).

| Modèle | Feature ablée | Classe cible | Images analysées | Prédictions avant | Prédictions après | Flipped |
|--------|---------------|:------------:|:----------------:|:-----------------:|:-----------------:|:-------:|
| DeiT-Base | Top feature par classe | class_i | 100 | 100 % class_i | 100 % class_i | **0 %** |
| DINOv2 | Top feature par classe | class_i | 100 | 100 % class_i | 100 % class_i | **0 %** |

> **Résultat :** L'ablation des features SAE ne modifie pas la prédiction pour aucun des deux modèles. Ce résultat est directement cohérent avec les fréquences d'activation quasi nulles : une feature qui s'active sur < 0.01 % à 0.69 % des tokens (DeiT) ou < 0.001 % à 13.69 % (DINOv2) n'est pas causalement responsable de la décision finale.

---

### 7.8 Discussion et limites

#### Diagnostic : pourquoi les features sont-elles peu fréquentes ?

| Cause probable | Explication |
|----------------|-------------|
| Sparsité trop élevée (k=32) | Avec 12 288 features et seulement 32 actives par token, chaque feature a statistiquement peu d'opportunités de s'activer ; les concepts sont dispersés sur de nombreuses features rares |
| Dictionnaire sur-dimensionné (16×) | L'expansion 16× peut dépasser la diversité sémantique réelle de la couche 9 ; une expansion 4× ou 8× forcerait une meilleure utilisation |
| Superposition de features | Une même région tissulaire active des features différentes selon les images, signe d'une décomposition instable |
| Couche 9 encore polysémantique | Malgré le fine-tuning, la couche 9 encode encore des features de bas niveau partagées entre classes |
| Corpus d'entraînement SAE | DeiT : 24 829 images × 196 patches ≈ 4,9 M tokens ; DINOv2 : [à compléter] |

#### Conséquences pour l'interprétabilité

- Il n'est **pas possible** d'associer une feature SAE à un concept histologique de façon robuste avec la configuration actuelle.
- L'ablation causale est **non informative** : aucun changement de prédiction ne peut être attribué à une feature individuelle.
- Les cartes spatiales d'activation sont visuellement localisées (cohérentes pour LYM et NORM) mais ne sont **pas statistiquement représentatives** d'une classe pour la majorité des tissus.

#### Pistes d'amélioration

| Piste | Description |
|-------|-------------|
| Augmenter k | k=64 ou k=128 pour obtenir des features plus fréquentes sans sacrifier la sparsité |
| Réduire le dictionnaire | Expansion 4× (3 072 features) ou 8× (6 144 features) plutôt que 16× |
| Entraîner sur la couche 10 | Tester si la couche 10 produit des features plus stables et fréquentes |
| Augmenter le corpus | Utiliser l'intégralité de NCT-CRC-HE-100K (100 000 images) pour l'extraction |
| SAE sur CLS (expérience complémentaire) | Vérifier si le token CLS produit des features plus fréquentes (au prix de la localisation spatiale) |

---

## Annexes

### A. Paramètres XAI utilisés

| Paramètre | Valeur | Applicable à |
|---|---|---|
| `FAITH_N_STEPS` | 50 | Insertion / Deletion |
| `INSERTION_BASELINE` | `black` | Insertion |
| `DELETION_REPLACEMENT` | `black` | Deletion |
| `N_IMAGES_PER_CLASS` | 50 | Toutes méthodes |
| GradCAM `aug_smooth` | `False` | GradCAM |
| GradCAM `eigen_smooth` | `False` | GradCAM |
| IG `n_steps` | 50 | Integrated Gradients |
| IG `internal_batch_size` | 8 (CNN) / 4 (ViT) | Integrated Gradients |
| IG `method` | `gausslegendre` | Integrated Gradients |
| IG `baseline` | `black_image` | Integrated Gradients |
| Rollout `head_fusion` | `mean` | Attention Rollout |
| Rollout `discard_ratio` | 0.9 | Attention Rollout |
| Rollout `residual_weight` | 0.5 | Attention Rollout |
| LRP `gamma` | 0.25 | LRP |
| LRP `epsilon` | 0.25 | LRP |

### B. Couches GradCAM par modèle

| Modèle | Couche cible | Résolution spatiale brute |
|---|---|---|
| ResNet-50 | `layer4[-1]` (dernier bloc résiduel) | 14×14 |
| ViT-Base/16 | `blocks.11.norm1` | 14×14 (patches) |
| DeiT-Base | `blocks.11.norm1` | 14×14 (patches) |
| DINOv2-ViT-B/14 | `backbone.blocks.11.norm1` | 16×16 (patches) |
| Swin-Base | `layers.3.blocks.1.norm2` | 7×7 (fenêtres stage 4) |

### C. Compatibilité méthodes × architectures

| Méthode | ResNet-50 | ViT-Base | DeiT-Base | DINOv2 | Swin-Base |
|---|---|---|---|---|---|
| GradCAM | ✓ | ✓ | ✓ | ✓ | ✓ |
| Integrated Gradients | ✓ | ✓ | ✓ | ✓ | ✓ |
| Attention Rollout | ✗ | ✓ | ✓ | ✓ | ✗ |
| Generic Attention ⭐ | ✗ | ✓ | ✓ | ✓ | ✗ |
| LRP | ✓ | ✓ | ✓ | ✓ | ✓ |


---
