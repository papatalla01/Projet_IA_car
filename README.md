🚗 Projet IA Embarquée – Véhicule Autonome
📌 Présentation générale

Ce projet s’inscrit dans le cadre du module IA embarquée et a pour objectif la conception d’un système autonome capable d’analyser son environnement et de prendre des décisions en temps réel.

Le projet aboutit à une compétition entre deux véhicules autonomes, évaluant les performances des algorithmes d’intelligence artificielle embarqués (vision, contrôle, perception, capteurs).

🎯 Objectifs

Concevoir une chaîne complète d’IA embarquée, de l’entraînement à l’inférence temps réel

Implémenter des algorithmes rapides et optimisés pour une plateforme embarquée

Exploiter plusieurs modalités :

Vision par caméra

Lidar

Commandes vocales

GPS / IMU

Apprentissage par renforcement

Respecter des contraintes matérielles et mémoire strictes

⚙️ Contraintes techniques
🔁 Workflow imposé
PyTorch (entraînement)
        ↓
      ONNX
        ↓
TensorRT (inférence embarquée)

🧠 Plateforme matérielle

Jetson Orin Nano

JetPack 6.1

Capteurs :

Caméra

Lidar

IMU

GPS

🚀 Performance

Algorithmes rapides et temps réel

Optimisations possibles via :

TensorRT avancé

CUDA custom (si nécessaire)

🧪 Projets et briques IA développées
🖼️ Vision par ordinateur

U-Net – Pattern Recognition

Segmentation d’éléments de l’environnement

U-Net 3D monoculaire

Estimation de profondeur à partir d’une caméra RGB

🎮 Apprentissage par renforcement (RL)

Contrôle du véhicule autonome

Prise de décision en fonction de l’environnement perçu

🎙️ Voice Learning

Reconnaissance de commandes vocales

Communication via Wi-Fi

Utilisation de datasets audio + augmentation

📡 Lidar Learning

Détection d’obstacles par intelligence artificielle

Traitement de nuages de points / scans lidar

🧭 GPS & IMU

Fusion de capteurs basée sur l’IA

Estimation de position et d’orientation du véhicule

🗂️ Données et apprentissage

Création et gestion de bases de données personnalisées

Augmentation de données (image, audio, capteurs)

Entraînement sur machine de calcul (ex. Google Colab)

Export des modèles vers ONNX puis TensorRT

🧠 Méthodologie R&D

Analyse et compréhension des algorithmes existants

Implémentation progressive et itérative

Validation sur PC puis déploiement embarqué

Optimisation mémoire et latence

Tests en conditions réelles sur véhicule

🏁 Résultat attendu

Un véhicule autonome fonctionnel

Des modèles IA embarqués optimisés

Une démonstration finale sous forme de compétition entre véhicules

📦 Technologies utilisées

Python / PyTorch

ONNX

TensorRT

CUDA

Linux embarqué (JetPack)

Traitement du signal, vision, IA
