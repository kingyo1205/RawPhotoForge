import { join } from "jsr:@std/path";
import { copy } from "jsr:@std/fs"

const DIST_DIR = "dist";
const ENTRY_HTML = "index.html";

async function initDist() {


    try {
        await Deno.remove(DIST_DIR, { recursive: true });
    } catch {

    }


    await Deno.mkdir(DIST_DIR, { recursive: true });
    await copy("assets", join(DIST_DIR, "assets"))
    await copy("style.css", join(DIST_DIR, "style.css"))
}

async function bundle() {
    console.log("\n[build] bundling...");
    await initDist();

    const command = new Deno.Command("deno", {
        args: [
            "bundle",
            "--unstable-raw-imports",
            "--platform",
            "browser",
            "--outdir",
            DIST_DIR,
            ENTRY_HTML,
        ],
        stdout: "inherit",
        stderr: "inherit",
    });

    return command.spawn();
}



const bundleProcess = await bundle();

await bundleProcess.status;
