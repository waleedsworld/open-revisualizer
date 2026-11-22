// Compress a 3DGS .ply to .spz (about 10x smaller) with Spark's transcoder, in a headless browser.
//   node tools/to_spz.mjs in.ply out.spz [--max-sh 2]
import { chromium } from "playwright";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";

const [inp, out, ...rest] = process.argv.slice(2);
const i = rest.indexOf("--max-sh");
const maxSh = i >= 0 ? +rest[i + 1] : 3;
const webDir = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../web");

// tiny static server: /vendor from web/, /input.ply from the argument, POST /out writes the result
const server = http.createServer((req, res) => {
  if (req.method === "POST" && req.url === "/out") {
    const ws = fs.createWriteStream(out);
    req.pipe(ws);
    ws.on("finish", () => res.end("ok"));
    return;
  }
  const file = req.url === "/input.ply" ? path.resolve(inp) : path.join(webDir, decodeURIComponent(req.url.split("?")[0]));
  if (req.url === "/") { res.setHeader("content-type", "text/html"); return res.end(PAGE); }
  if (!fs.existsSync(file)) { res.statusCode = 404; return res.end(); }
  if (file.endsWith(".js")) res.setHeader("content-type", "text/javascript");
  fs.createReadStream(file).pipe(res);
});
const PAGE = `<!doctype html><script type="importmap">{"imports":{"three":"/vendor/three.module.js","three/addons/":"/vendor/addons/","@sparkjsdev/spark":"/vendor/spark.module.js"}}</script>
<script type="module">
import { transcodeSpz } from "@sparkjsdev/spark";
const bytes = new Uint8Array(await (await fetch("/input.ply")).arrayBuffer());
const { fileBytes, clippedCount } = await transcodeSpz({ inputs: [{ fileBytes: bytes, fileType: "ply", pathOrUrl: "input.ply" }], maxSh: ${maxSh} });
await fetch("/out", { method: "POST", body: fileBytes });
window.done = { size: fileBytes.length, clippedCount };
</script>`;
await new Promise((r) => server.listen(0, r));
const port = server.address().port;
const browser = await chromium.launch();
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
await page.goto(`http://127.0.0.1:${port}/`);
const done = await page.waitForFunction(() => window.done, null, { timeout: 30 * 60 * 1000 });
console.log("wrote", out, await done.jsonValue());
await browser.close();
server.close();
