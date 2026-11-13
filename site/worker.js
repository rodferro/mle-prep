// Runs Python + pytest in a Web Worker so an infinite loop can't freeze the page.
const PYODIDE_VERSION = "0.28.0";
importScripts(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`);

const SETUP = String.raw`
import os, sys, io, importlib, contextlib
sys.dont_write_bytecode = True
os.environ.setdefault("USER", "pyodide")   # pytest's tmp_path needs a user name
WORK = "/home/pyodide/mle"

def write_file(path, text):
    full = os.path.join(WORK, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(text)

def run_tests(test_file, use_solutions):
    import pytest
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in ("problems", "solutions", "conftest") or root.startswith("test_"):
            del sys.modules[name]
    importlib.invalidate_caches()
    if use_solutions:
        os.environ["USE_SOLUTIONS"] = "1"
    else:
        os.environ.pop("USE_SOLUTIONS", None)
    os.chdir(WORK)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = pytest.main([
            "-q", "--color=no", "--tb=short",
            "-p", "no:cacheprovider", "-p", "no:faulthandler",
            "tests/" + test_file,
        ])
    return int(code), buf.getvalue()
`;

let pyodide;

const ready = (async () => {
  pyodide = await loadPyodide();
  try {
    await pyodide.loadPackage("pytest");
  } catch (_) {
    await pyodide.loadPackage("micropip");
    await pyodide.runPythonAsync("import micropip; await micropip.install('pytest')");
  }
  pyodide.runPython(SETUP);
  postMessage({ type: "ready" });
})().catch((err) => postMessage({ type: "fatal", error: String(err) }));

onmessage = async (event) => {
  await ready;
  const { id, files, testFile, useSolutions } = event.data;
  try {
    const write = pyodide.globals.get("write_file");
    for (const [path, text] of Object.entries(files)) write(path, text);
    write.destroy();

    const run = pyodide.globals.get("run_tests");
    const result = run(testFile, !!useSolutions);
    const [code, output] = result.toJs();
    result.destroy();
    run.destroy();
    postMessage({ type: "result", id, code, output });
  } catch (err) {
    postMessage({ type: "result", id, code: -1, output: String(err) });
  }
};
