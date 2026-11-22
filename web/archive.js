// A cached answer is keyed by the complete question, including the queried sender or opponent.
let dataset;
let card;
async function asset(name) {
  const response = await fetch("./" + name);
  if (!response.ok) throw Error("The packaged demo asset is unavailable: " + name);
  return response.json();
}
export async function staticRequest(path, body) {
  if (path === "evaluation") {
    card ||= asset("model-card.json");
    return (await card).evaluation;
  }
  dataset ||= asset("historical-demo.json");
  const archive = await dataset;
  const query = new URL(path,"https://marketflow.invalid/");
  const dates = Object.keys(archive.days);
  const date = body?.date || query.searchParams.get("date") || dates[0];
  const day = archive.days[date];
  if (!day) throw Error("Choose one of the dates in the packaged replay.");
  if (query.pathname === "/catalog") return {date,dates,targets:Object.keys(day),dataset_version:archive.dataset_version};
  const target = body?.target || query.searchParams.get("target");
  const record = day[target];
  if (!record) throw Error("Choose a stock available for this archived date.");
  if (query.pathname === "/snapshot") return record.snapshot;
  if (query.pathname === "/intervene" && record.interventions[body.sender]) return record.interventions[body.sender];
  if (query.pathname === "/copilot") {
    const key = body.task === "COUNTERFACTUAL_COT" ? body.sender : body.task === "PAIRWISE_FORECAST_COT" ? body.other : "";
    const result = record.answers[body.task]?.[key];
    if (result) return {...result,grounding_gate:archive.gate,cached:true,task:body.task,model_id:"DRIFT-TSLM-v2.1-Qwen3.5-4B",dataset_version:archive.dataset_version,sample_id:"SP500|"+date+"|"+target};
  }
  throw Error("This comparison is unavailable in the packaged replay. Choose a listed stock or upstream edge.");
}
