# MarketFlow demo

From the repository root, run `npm --prefix web run build` and `python3 -m http.server 8000 --directory web/dist`. Open `http://127.0.0.1:8000`. Run `npm --prefix web run pack` to regenerate the ZIP from the selected checkpoint’s exported answers. The ZIP in `reports/marketflow-static.zip` contains the same static files and can also be served from its extracted root.

On Vercel select `web` as the root directory. The build command is `npm run build` and the output directory is `dist`. The packaged replay has three stocks on each of six dates. Every displayed incoming edge has an exact DRIFT intervention and a precomputed Qwen answer. The other two stocks can each be used as the forecast opponent. “Archived inference” identifies these outputs.

For online inference on the configured training host, run `bash web/serve.sh`. It reads `data/prompt_data/local/history-v3` and the validation-selected weights in `checkpoints/TSLM_training/v2.1/drift`. For separate hosting set `PUBLIC_MARKETFLOW_API_URL` during the web build and configure `MARKETFLOW_CORS_ORIGINS` on the backend. Connection failure activates the packaged replay.

`GET /api/catalog` lists dates and stocks. `GET /api/snapshot` reads the observation. `POST /api/intervene` takes date, target and sender. `POST /api/copilot` takes date, target and task plus a sender for edge masking or a second stock for pairwise prediction. `GET /api/evaluation` serves the held-out model card.

To reproduce this repair, run `.venv/bin/python code/TSLM_training/v2_features.py`, then `.venv/bin/python code/TSLM_training/v2_run_comparisons.py` on the configured GPU host. The latter trains the main model, runs its ablations and pairwise readout, and uses the archived comparison predictions. Run `export_v2_demo.py` in the same GPU environment as the API to regenerate the archived inference from the selected checkpoint.
