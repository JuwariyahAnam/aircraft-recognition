# 3D aircraft models

`registry.json` lists the realistic models the demo can show. Each entry has:

* `file`: the `.glb` file in this folder
* `name`: the name shown on screen
* `class`: which of the four types it belongs to (used when no exact match is known)
* `matches`: aircraft names from the fine-grained classifier that should show this model
* `spin` (optional): mesh names that rotate (rotors, propellers): `{"MeshName": ["y", speed]}`
* `credit` / `source`: the author and page, for your report's credits

Types with no model here fall back to the aircraft drawn in code (`viewer3d.js`).

## Adding a model

1. Download a model (e.g. from Sketchfab, glTF/GLB format, CC licence).
2. Make sure its nose points along +X and its top along +Y, centred, about 4.4 units long.
   Models converted with this project's converter are already prepared this way.
3. Copy the `.glb` here and add an entry to `registry.json`.

## Credits for the included models

* Boeing 737 AEW&C Prototype by Muhamad Mirza Arrafi:
  https://sketchfab.com/3d-models/boeing-737-aewc-prototype-bd2d73c132d14a609dbf13bd4cdc652e
  (the Sketchfab page is labelled "Generated with AI")
* General Dynamics F-16D Block 60 by Muhamad Mirza Arrafi:
  https://sketchfab.com/3d-models/general-dynamics-f-16d-block-60-ea8edb08d79e4eafa9c794c581f88cf3
* SH-60B Seahawk Helicopter by Muhamad Mirza Arrafi, Creative Commons Attribution (CC BY 4.0):
  https://sketchfab.com/3d-models/sh-60b-seahawk-helicopter-558373489bc145e292342efbe4b982ed

The OBJ files had no material files, so textures were matched to parts by hand,
the models were turned to face +X, scaled, and saved as GLB with embedded textures.
