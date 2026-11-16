#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "src"))
from load_checkpoint import load_model
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/"MANIFEST.json").read_text())
    checked=0
    for rel, expected in manifest["file_sha256"].items():
        assert sha(ROOT/rel)==expected, f"Hash mismatch: {rel}"
        checked+=1
    rows=[]
    for item in manifest["models"]:
        model, config, ck=load_model(item["cell"])
        torch.manual_seed(7)
        with torch.no_grad():
            output=model(torch.randn(8, int(ck["lookback"]), int(ck["feature_dim"])), compute_graph_diagnostics=False)
        assert output.prediction.shape==(8,) and torch.isfinite(output.prediction).all()
        assert Path(config["data"]["provider_path"]).is_dir()
        assert Path(config["model_config"]).is_file()
        rows.append({"cell":item["cell"], "strict_state_load":True,"finite_forward":True,
            "feature_dim":int(ck["feature_dim"]),"epoch":int(ck["epoch"])})
    import qlib
    from qlib.data import D
    data_results=[]
    for entry in manifest["datasets"]:
        provider=ROOT/entry["path"]
        files=[f for f in provider.rglob("*") if f.is_file()]
        assert len(files)==entry["file_count"], (len(files),entry["file_count"])
        assert sum(f.stat().st_size for f in files)==entry["file_bytes"]
        qlib.init(provider_uri=str(provider), region=entry["region"], kernels=1,
                  expression_cache=None,dataset_cache=None)
        for market in entry["markets"]:
            members=[]
            for line in (provider/"instruments"/(market+".txt")).read_text().splitlines():
                parts=line.split()
                if len(parts)>=3 and parts[1]<="2025-12-31" and parts[2]>="2025-12-31": members.append(parts[0])
            assert len(members)>=2
            frame=D.features(members[:2], ["$close","$open","$volume","Ref($close,-2)/Ref($close,-1)-1"],
                             start_time="2025-12-31",end_time="2025-12-31",freq="day")
            assert len(frame)>0 and frame.iloc[:,-1].notna().any()
            data_results.append({"dataset":entry["id"],"market":market,"sample_rows":len(frame),
                                 "raw_fields_and_label_readable":True})
    for name, module in tuple(sys.modules.items()):
        if name.startswith("meta_fdg") and getattr(module,"__file__",None):
            assert Path(module.__file__).resolve().is_relative_to((ROOT/"code/src").resolve()), name
    result={"status":"pass","hashed_files":checked,"models":rows,"data":data_results,
            "full_test_metrics_recomputed":False,"training_started":False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
