# Claude (claude.ai Projects)

Prefer the skill: zip the `skills/video-to-physics/` folder and upload it under **Settings > Capabilities > Skills**.
Claude then loads `SKILL.md`, the references and the tools on demand. Its code environment has no Blender and no
solvers, so Claude writes scripts there and you run them locally.

Without skills, use a Project:

1. Paste `adapters/universal-system-prompt.md` (the part below the line) into the project instructions.
2. Add to project knowledge: `SKILL.md`, `references/workflow.md`, `references/physics-routing.md`,
   `references/verification.md`, `references/blender-integration.md` and `templates/CONTRACT.md`.

For Claude Code, install the plugin or copy the skill folder (see the repository README). Claude Code can run the
tools, the solvers and Blender itself, and can run the specialist roles as parallel sub-agents.
