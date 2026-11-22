# Add the DRIFT Alpha158 stock-flow connectors

The public connector downloads a pinned price archive from Kaggle without credentials. It recomputes 157 named factors, preserves the 20 trading dates per signal and attaches four tasks per record. These tasks ask about signal attribution, a masked incoming edge, the target’s role and regime, and the realized winner of a stock pair. The dates split into 2018–2022 training, 2023 validation and 2024–2025 test.

**Reproduce the public release**

```bash
timenet-build build 10eit/drift-alpha158-flow-public --local-registry .timenet-registry
python -c "from timenet import TimeNet; d=TimeNet(sources=['.timenet-registry']).load('10eit/drift-alpha158-flow-public@1.0.0'); print(len(d.records))"
```

The default isolated public build used Python 3.12, NumPy 2.5.5 and pandas 2.3.3. It matched the local-bundle replay at every signal value and task. The result has 98 records from 30 target companies in 11 sectors, with 156 / 68 / 168 tasks by split. Independent online and offline builds matched bit-for-bit across 678,160 signal values and exactly across all 392 prompts and answers. The online build downloaded 266,011,984 bytes. Its registry occupied 5,741,180 bytes on this host.

The public records are the overlap with an archived 240-record DRIFT research release. All seven stocks in every public record matched the historical daily returns within 1.12 × 10⁻⁶ and the independently calculated pairwise outcome agreed in every case. Annotations identify the two providers. Model interventions describe a model score change.

The bundled index contains observation IDs, dates, stock names and task annotations. Prices are downloaded at build time. Kaggle lists the underlying snapshot as CC0. The data card specifies the combined release license as unspecified. The original provider’s 240-record edition remains a separate restricted dataset with an explicit local-bundle or verified HTTPS-mirror input. [Price source and version](https://www.kaggle.com/datasets/joebeachcapital/s-and-p500-index-stocks-daily-updated/versions/613)

**Verification against the upstream tree** · The patch applies cleanly to `979a0ae`. `make check` (schema generation, Ruff formatting, lint and ty) passed. `make test` completed with 2,986 passed, one skipped and two expected failures. `make test-connectors` passed for every connector, including both DRIFT editions with the actual local archives available.
