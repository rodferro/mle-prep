(() => {
  const $ = (id) => document.getElementById(id);
  const state = { data: null, current: null, editor: null, worker: null, ready: false, runId: 0 };

  // ---------- Python worker ----------
  function startWorker() {
    state.ready = false;
    setStatus("loading", "Loading Python…");
    setRunEnabled(false);
    state.worker = new Worker("worker.js");
    state.worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "ready") {
        state.ready = true;
        setStatus("ready", "Python ready");
        setRunEnabled(true);
        preload(state.current);
      } else if (msg.type === "status") {
        if (msg.error) {
          setStatus("error", "Package failed to load");
          $("output").textContent = msg.error;
        } else if (msg.text) {
          setStatus("loading", msg.text);
        } else if (state.ready) {
          setStatus("ready", "Python ready");
        }
      } else if (msg.type === "fatal") {
        setStatus("error", "Python failed to load");
        $("output").textContent =
          "Could not load Pyodide from cdn.jsdelivr.net.\n" +
          "If you're on a corporate network, it may be blocked.\n\n" + msg.error;
      } else if (msg.type === "result" && msg.id === state.runId) {
        showResult(msg.code, msg.output);
      }
    };
    state.worker.onerror = (e) => {
      setStatus("error", "Python error");
      $("output").textContent = String(e.message || e);
    };
  }

  function setStatus(kind, text) {
    const el = $("py-status");
    el.className = "status " + kind;
    el.textContent = text;
  }

  function setRunEnabled(on) {
    $("run").disabled = !on;
    $("run-reference").disabled = !on;
  }

  function run(useSolutions) {
    const p = state.current;
    if (!p || !state.ready) return;
    const files = {
      "problems/__init__.py": "",
      "solutions/__init__.py": "",
      "tests/conftest.py": state.data.conftest,
      ["tests/" + p.test_file]: p.tests,
      ["problems/" + p.module + ".py"]: state.editor.getValue(),
      ["solutions/" + p.module + ".py"]: p.solution,
    };
    state.runId += 1;
    $("result-banner").hidden = true;
    $("output").textContent = useSolutions ? "Running tests on the reference solution…" : "Running tests…";
    $("run").hidden = true;
    $("stop").hidden = false;
    state.worker.postMessage({
      type: "run",
      id: state.runId,
      files,
      testFile: p.test_file,
      useSolutions,
      packages: p.packages || [],
    });
  }

  // Start downloading a problem's extra packages (e.g. pandas) as soon as it's opened.
  function preload(p) {
    if (p && p.packages && p.packages.length && state.worker) {
      state.worker.postMessage({ type: "preload", packages: p.packages });
    }
  }

  function showResult(code, output) {
    $("run").hidden = false;
    $("stop").hidden = true;
    const banner = $("result-banner");
    const summary = (output.trim().split("\n").pop() || "").replace(/=+/g, "").trim();
    banner.hidden = false;
    if (code === 0) {
      banner.className = "pass";
      banner.textContent = "✓ All tests passed — " + summary;
    } else {
      banner.className = "fail";
      banner.textContent = "✗ " + (summary || "Tests failed");
    }
    $("output").textContent = output;
  }

  function stop() {
    state.worker.terminate();
    state.runId += 1;
    $("run").hidden = false;
    $("stop").hidden = true;
    $("result-banner").hidden = false;
    $("result-banner").className = "fail";
    $("result-banner").textContent = "■ Stopped (restarting Python…)";
    startWorker();
  }

  // ---------- views ----------
  function renderList() {
    const rows = state.data.problems.map((p) => `
      <tr data-id="${p.id}">
        <td>${p.id.toUpperCase()}</td>
        <td><a href="#${p.id}">${escapeHtml(p.title)}</a></td>
        <td class="diff-${p.difficulty}">${p.difficulty}</td>
        <td>${p.tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</td>
      </tr>`).join("");
    $("problem-rows").innerHTML = rows;
    $("problem-rows").querySelectorAll("tr").forEach((tr) => {
      tr.addEventListener("click", () => { location.hash = tr.dataset.id; });
    });
  }

  function showProblem(p) {
    state.current = p;
    document.title = `${p.id.toUpperCase()} · ${p.title} — MLE Prep`;
    $("p-title").textContent = `${p.id.toUpperCase()}. ${p.title}`;
    $("p-meta").innerHTML = `<span class="diff-${p.difficulty}">${p.difficulty}</span> &nbsp; ` +
      p.tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("");
    $("p-statement").textContent = p.statement;
    $("solution").textContent = p.solution;
    $("solution-panel").hidden = true;
    $("toggle-solution").textContent = "Show solution";
    $("result-banner").hidden = true;
    $("output").textContent = "Press “Run tests” (or Ctrl/Cmd + Enter) to check your solution.";
    $("list-view").hidden = true;
    $("problem-view").hidden = false;
    state.editor.setValue(p.stub);
    preload(p);
    state.editor.clearHistory();
    state.editor.refresh();
    state.editor.focus();
  }

  function route() {
    const id = location.hash.replace(/^#/, "").toLowerCase();   // Anki links use "#P03"
    const p = state.data.problems.find((x) => x.id === id);
    if (p) {
      showProblem(p);
    } else {
      state.current = null;
      document.title = "MLE Prep";
      $("problem-view").hidden = true;
      $("list-view").hidden = false;
    }
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------- init ----------
  async function init() {
    state.editor = CodeMirror($("editor"), {
      mode: "python",
      theme: "material-darker",
      lineNumbers: true,
      indentUnit: 4,
      matchBrackets: true,
      extraKeys: {
        Tab: (cm) => (cm.somethingSelected() ? cm.indentSelection("add") : cm.execCommand("insertSoftTab")),
        "Shift-Tab": (cm) => cm.indentSelection("subtract"),
        "Ctrl-Enter": () => run(false),
        "Cmd-Enter": () => run(false),
      },
    });

    $("run").addEventListener("click", () => run(false));
    $("run-reference").addEventListener("click", () => run(true));
    $("stop").addEventListener("click", stop);
    $("reset").addEventListener("click", () => {
      if (state.current && confirmReset()) state.editor.setValue(state.current.stub);
    });
    $("toggle-solution").addEventListener("click", () => {
      const panel = $("solution-panel");
      panel.hidden = !panel.hidden;
      $("toggle-solution").textContent = panel.hidden ? "Show solution" : "Hide solution";
      state.editor.refresh();
    });

    startWorker();

    const res = await fetch("problems.json", { cache: "no-cache" });
    state.data = await res.json();
    renderList();
    window.addEventListener("hashchange", route);
    route();
  }

  function confirmReset() {
    const btn = $("reset");
    if (btn.dataset.armed) {
      delete btn.dataset.armed;
      btn.textContent = "Reset";
      return true;
    }
    btn.dataset.armed = "1";
    btn.textContent = "Click again to reset";
    setTimeout(() => { delete btn.dataset.armed; btn.textContent = "Reset"; }, 2500);
    return false;
  }

  init();
})();
