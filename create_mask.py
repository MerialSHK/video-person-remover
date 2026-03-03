#!/usr/bin/env python3
"""
Masken-Generator für Wasserzeichen-Entfernung.
Erstellt eine schwarze Maske mit weissem Rechteck an der Logo-Position.
"""

import argparse
from PIL import Image, ImageDraw


def create_mask(width: int, height: int, x: int, y: int, w: int, h: int, output: str, padding: int = 0):
    """
    Erstellt eine Maske für Inpainting.

    Args:
        width: Bildbreite
        height: Bildhöhe
        x: X-Position des Wasserzeichens (links oben)
        y: Y-Position des Wasserzeichens (links oben)
        w: Breite des Wasserzeichens
        h: Höhe des Wasserzeichens
        output: Ausgabepfad
        padding: Zusätzlicher Rand um das Wasserzeichen
    """
    # Schwarzes Bild erstellen
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Weisses Rechteck für Wasserzeichen-Bereich (mit Padding)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    draw.rectangle([x1, y1, x2, y2], fill=255)

    mask.save(output)
    print(f"Maske erstellt: {output}")
    print(f"  Bildgrösse: {width}x{height}")
    print(f"  Maskenbereich: ({x1}, {y1}) bis ({x2}, {y2})")
    print(f"  Maskengrösse: {x2-x1}x{y2-y1} Pixel")


def main():
    parser = argparse.ArgumentParser(
        description='Erstellt eine Maske für Wasserzeichen-Entfernung',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Maske für Veo-Logo unten rechts
  python3 create_mask.py -W 4608 -H 2612 -x 4200 -y 2500 -w 350 --wm-height 80 -o mask.png

  # Mit zusätzlichem Padding
  python3 create_mask.py -W 4608 -H 2612 -x 4200 -y 2500 -w 350 --wm-height 80 -p 10 -o mask.png

  # Ersten Frame extrahieren zur Vermessung:
  ffmpeg -i video.mov -vframes 1 first_frame.png
        """
    )

    parser.add_argument('-W', '--width', type=int, default=4608,
                        help='Bildbreite (default: 4608)')
    parser.add_argument('-H', '--height', type=int, default=2612,
                        help='Bildhöhe (default: 2612)')
    parser.add_argument('-x', type=int, required=True,
                        help='X-Position des Wasserzeichens (links)')
    parser.add_argument('-y', type=int, required=True,
                        help='Y-Position des Wasserzeichens (oben)')
    parser.add_argument('-w', '--wm-width', type=int, required=True,
                        help='Breite des Wasserzeichens')
    parser.add_argument('--wm-height', type=int, required=True,
                        help='Höhe des Wasserzeichens')
    parser.add_argument('-p', '--padding', type=int, default=5,
                        help='Zusätzlicher Rand um das Wasserzeichen (default: 5)')
    parser.add_argument('-o', '--output', type=str, default='mask.png',
                        help='Ausgabedatei (default: mask.png)')

    args = parser.parse_args()

    create_mask(
        width=args.width,
        height=args.height,
        x=args.x,
        y=args.y,
        w=args.wm_width,
        h=args.wm_height,
        output=args.output,
        padding=args.padding
    )


if __name__ == '__main__':
    main()
