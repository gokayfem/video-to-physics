# ChatGPT (Custom GPT)

1. Create a GPT and paste `adapters/universal-system-prompt.md` (the part below the line) into **Instructions**.
2. Upload as **Knowledge**: `skills/video-to-physics/SKILL.md` and the files in `skills/video-to-physics/references/`.
3. Enable **Code Interpreter** so it can run the plain-Python parts of `scripts/v2p/` (measurement, checks,
   elastica, dye). Blender and the CFD/FEM solvers run on your machine.
