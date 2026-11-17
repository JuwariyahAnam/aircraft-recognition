# Optional: realistic 3D model files

The demo draws its own aircraft by default. To use a professionally made
model instead, put a glTF binary file here with one of these names:

    passenger_jet.glb   fighter_jet.glb   helicopter.glb   propeller.glb

The viewer finds it automatically, centres it and scales it to fit.

If the nose doesn't point to the right, add a JSON file with the same name,
e.g. `passenger_jet.json`:

    {"rotateX": 0, "rotateY": 90, "rotateZ": 0}

(degrees; try 90, 180 or -90 on rotateY first).

Where to get free models: search Sketchfab (sketchfab.com) with the
"Downloadable" filter and a Creative Commons licence, and download in
glTF/GLB format. Keep the files small (under ~20 MB) so the page loads
quickly. Credit the author in your report if the licence is CC-BY.
