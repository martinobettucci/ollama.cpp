#!/usr/bin/env python3
"""Génère des images PNG de test, sans dépendance externe.

@spec docs/BACKLOG.md OC-032 « Détection de capacités observables », OC-044 « /api/chat »
@spec docs/ollama.cpp-architecture.md §5.4 « Modèle canonique » (entrées image)
@spec docs/DAT.md §13 « Données de développement »

Vérifier la capacité `vision` demande une image **réelle** : la mission (§13) interdit d'annoncer
une capacité qui n'a pas été observée, et la présence d'un fichier `mmproj` dans un dépôt n'est
pas une observation — c'est une promesse. Seule une image effectivement décrite par le modèle
prouve que la chaîne image → projecteur → modèle fonctionne.

Ces images sont donc **délibérément triviales** : un aplat de couleur, ou une forme géométrique
franche sur fond blanc. Un modèle de vision, même très petit, doit les décrire correctement ; une
image complexe rendrait l'échec ambigu — incapacité du modèle, ou chaîne rompue ?

L'encodage PNG est écrit ici plutôt que délégué à Pillow : la dépendance n'est requise nulle part
ailleurs dans le projet, et un encodeur de quelques lignes évite de l'ajouter pour des tests.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

#: Palette de la charte P2Enjoy, pour que les données de test restent cohérentes avec le reste du
#: projet. Toutes ces teintes sont franches et sans ambiguïté de nom, ce qui est la seule
#: propriété dont dépend le test.
COULEURS: dict[str, tuple[int, int, int]] = {
    "bleu": (0x23, 0x46, 0x8C),
    "vert": (0x23, 0x8C, 0x33),
    "jaune": (0xD9, 0xCF, 0x4A),
    "rouge": (0xF2, 0x41, 0x41),
    "noir": (0x0D, 0x0D, 0x0D),
    "blanc": (0xFF, 0xFF, 0xFF),
}


def encoder_png(pixels: list[list[tuple[int, int, int]]]) -> bytes:
    """Encode une matrice RGB en PNG sans perte, format minimal (`RFC 2083`)."""
    hauteur = len(pixels)
    largeur = len(pixels[0])

    brut = bytearray()
    for ligne in pixels:
        brut.append(0)  # type de filtre 0 : aucun
        for rouge, vert, bleu in ligne:
            brut += bytes((rouge, vert, bleu))

    def bloc(nom: bytes, donnees: bytes) -> bytes:
        corps = nom + donnees
        return struct.pack(">I", len(donnees)) + corps + struct.pack(">I", zlib.crc32(corps))

    entete = struct.pack(">IIBBBBB", largeur, hauteur, 8, 2, 0, 0, 0)  # 8 bits, RGB
    return (b"\x89PNG\r\n\x1a\n"
            + bloc(b"IHDR", entete)
            + bloc(b"IDAT", zlib.compress(bytes(brut), 9))
            + bloc(b"IEND", b""))


def aplat(couleur: str, taille: int) -> bytes:
    """Image d'une seule couleur : le test le moins ambigu qui soit."""
    rgb = COULEURS[couleur]
    return encoder_png([[rgb] * taille for _ in range(taille)])


def disque(couleur: str, taille: int) -> bytes:
    """Disque plein centré sur fond blanc : couleur **et** forme sont vérifiables."""
    rgb = COULEURS[couleur]
    blanc = COULEURS["blanc"]
    centre = (taille - 1) / 2
    rayon = taille * 0.38
    pixels = []
    for y in range(taille):
        ligne = []
        for x in range(taille):
            dedans = (x - centre) ** 2 + (y - centre) ** 2 <= rayon**2
            ligne.append(rgb if dedans else blanc)
        pixels.append(ligne)
    return encoder_png(pixels)


def carre(couleur: str, taille: int) -> bytes:
    """Carré plein centré sur fond blanc, pour distinguer la forme du disque."""
    rgb = COULEURS[couleur]
    blanc = COULEURS["blanc"]
    marge = taille // 4
    pixels = []
    for y in range(taille):
        ligne = []
        for x in range(taille):
            dedans = marge <= x < taille - marge and marge <= y < taille - marge
            ligne.append(rgb if dedans else blanc)
        pixels.append(ligne)
    return encoder_png(pixels)


FORMES = {"aplat": aplat, "disque": disque, "carre": carre}


def main(argv: list[str] | None = None) -> int:
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("--output", required=True, type=Path, help="chemin du PNG à écrire")
    analyseur.add_argument("--forme", choices=sorted(FORMES), default="disque")
    analyseur.add_argument("--couleur", choices=sorted(COULEURS), default="rouge")
    analyseur.add_argument("--taille", type=int, default=224,
                           help="côté en pixels (224 : entrée usuelle des encodeurs d'image)")
    arguments = analyseur.parse_args(argv)

    if arguments.taille < 8:
        analyseur.error("une taille inférieure à 8 pixels n'a pas de sens")

    donnees = FORMES[arguments.forme](arguments.couleur, arguments.taille)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(donnees)
    print(f"{arguments.output} : {arguments.forme} {arguments.couleur} "
          f"{arguments.taille}×{arguments.taille}, {len(donnees)} octets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
