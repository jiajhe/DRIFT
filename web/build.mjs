import { copyFile, mkdir, mkdtemp, rename, writeFile, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const source = fileURLToPath(new URL(".", import.meta.url));
const output = join(source, "dist");
const endpoint = (process.env.PUBLIC_MARKETFLOW_API_URL || "").trim();
if (endpoint) {
  const url = new URL(endpoint);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error("PUBLIC_MARKETFLOW_API_URL must be an HTTP origin without credentials, query, or fragment");
  }
}

await mkdir(output, {recursive: true});
const staging = await mkdtemp(join(output, ".building-"));
const files = ["app.js", "archive.js", "multiday.js", "multiday-replay.json", "starmap.js", "starmap-demo.json", "style.css", "model-card.json", "historical-demo.json", "index.html"];
try {
  await Promise.all(files.map(name => copyFile(join(source, name), join(staging, name))));
  await writeFile(join(staging, "config.js"), "window.MARKETFLOW_API = " + JSON.stringify(endpoint) + ";\n");
  // Replace each completed artifact. Readers can finish requests against the previous file.
  for (const name of ["config.js", ...files]) {
    await rename(join(staging, name), join(output, name));
  }
  await rm(join(output, "public-treasury.json"), {force: true});
} finally {
  await rm(staging, {recursive: true, force: true});
}
