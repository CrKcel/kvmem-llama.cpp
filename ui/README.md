# Lightweight chat UI

This is a small Svelte entry point for the pinned llama.cpp UI. It reuses the upstream Markdown/code renderer (including sandboxed HTML preview), input components, SSE record helpers, and Dexie conversation storage. It does not initialize the upstream router/agent/MCP/stream-resumption lifecycle.

Build from the repository root with Node.js 22 and npm:

```bash
python3 scripts/build-webui.py
```

The script exports the pinned `llama.cpp/tools/ui` into ignored `build-ui/`, applies the local entry points there, checks types, and writes static files to `build/share/kvmem/ui/`. Source archives without Git metadata use their bundled UI source. The llama.cpp working tree is not modified. UI history/settings use a separate `kvmem-chat` browser namespace.

Rebuild the server, start the usual IQ3/IQ4 recipe, and open `http://127.0.0.1:18200/`. No Node.js is needed on machines running a package that includes the built page. `--ui-dir PATH` overrides the static directory and `--no-ui` disables it. `scripts/prebuilt/package.py --ui-dir PATH` can include separately built assets.

The settings panel leaves unmodified parameters out of requests. Template reasoning effort and thinking-token budget are separate. The output limit includes reasoning. Histories are browser-local; normal continuation sends the complete history without `cache_reset`. The page does not execute tools. Stop closes the request; reloading does not resume an interrupted stream.

The speed displayed is generated tokens divided by the server's generation duration, including reasoning tokens but excluding prefill and post-generation cache commits. Very short replies are not throughput benchmarks.

## Regression checks

Run against a disposable test service, not an active conversation. The HTTP check needs a model that can answer simple arithmetic, with a generation reserve of at least 128. The browser check uses up to 512 generated tokens and requires Playwright's Chromium or `CHROMIUM` pointing to a compatible installation.

```bash
KVMEM_UI_TEST_URL=http://127.0.0.1:18400 python3 scripts/test_webui.py
KVMEM_UI_TEST_URL=http://127.0.0.1:18400 node ui/test-browser.mjs
python3 scripts/test_start_server.py
```

Browser tests check real generation, local history after reload, code rendering, stop/continuation, and the absence of unsupported backend requests. Local image upload/follow-up was also checked on IQ3 27B with GPU Q8 vision and MTP3/ReplaySSM.
