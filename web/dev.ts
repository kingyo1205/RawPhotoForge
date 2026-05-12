
import { join, extname } from "jsr:@std/path";
import { copy } from "jsr:@std/fs"

const DIST_DIR = "dist";
const ENTRY_HTML = "index.html";
const PORT = 8000;

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
            "--watch",
            "--outdir",
            DIST_DIR,
            ENTRY_HTML,
        ],
        stdout: "inherit",
        stderr: "inherit",
    });

    return command.spawn();
}

function getContentType(path: string): string {
    const ext = extname(path);

    switch (ext) {
        case ".html":
            return "text/html; charset=utf-8";
        case ".js":
            return "application/javascript; charset=utf-8";
        case ".css":
            return "text/css; charset=utf-8";
        case ".json":
            return "application/json; charset=utf-8";
        case ".png":
            return "image/png";
        case ".jpg":
        case ".jpeg":
            return "image/jpeg";
        case ".svg":
            return "image/svg+xml";
        case ".webp":
            return "image/webp";
        case ".wasm":
            return "application/wasm";
        default:
            return "application/octet-stream";
    }
}

async function startServer() {
    console.log(`[server] http://localhost:${PORT}`);

    await Deno.serve({ port: PORT }, async (req) => {
        const url = new URL(req.url);

        let pathname = decodeURIComponent(url.pathname);

        if (pathname === "/") {
            pathname = "/index.html";
        }

        const filePath = join(DIST_DIR, pathname);

        try {
            const file = await Deno.readFile(filePath);

            return new Response(file, {
                headers: {
                    "content-type": getContentType(filePath),
                    "cache-control": "no-cache",
                },
            });
        } catch {
            return new Response("404 Not Found", {
                status: 404,
            });
        }
    });
}

const bundleProcess = await bundle();

await startServer();

await bundleProcess.status;

