# Published static bundle

This is the exact public browser experience, including the certified network and saved training evidence. It requires no server-side runtime or connection to the original Mac.

Regenerate from the project root:

```bash
python3 build_site.py --run-dir runs/20261113T233235Z-7e5be6 --output site/dist
```

Serve locally with `python3 -m http.server 8000 --directory site/dist`.

The Sites identity is public metadata in `.openai/hosting.json`; no source credential or secret is included.
