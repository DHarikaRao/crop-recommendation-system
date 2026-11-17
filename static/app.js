(function () {
  "use strict";
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function profiles() {
    var el = document.getElementById("profiles");
    return el ? JSON.parse(el.textContent) : {};
  }

  /* ---- sliders stay in sync with the number inputs ---- */
  function bindField(field) {
    var number = field.querySelector('input[type="number"]');
    var range = field.querySelector('input[type="range"]');
    if (!number || !range) return;

    function paint() {
      var min = parseFloat(range.min), max = parseFloat(range.max);
      var value = parseFloat(range.value);
      var pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
      range.style.setProperty("--fill", pct + "%");  // CSSOM, allowed by the strict CSP
    }
    range.addEventListener("input", function () { number.value = range.value; paint(); });
    number.addEventListener("input", function () {
      if (number.value !== "" && !isNaN(parseFloat(number.value))) range.value = number.value;
      paint();
    });
    if (number.value !== "") range.value = number.value;
    paint();
  }

  function setValues(form, values) {
    Object.keys(values).forEach(function (name) {
      var input = form.querySelector('input[name="' + name + '"]');
      if (!input) return;
      input.value = values[name];
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }

  function curlFor(form) {
    var body = {};
    new FormData(form).forEach(function (value, key) { body[key] = Number(value); });
    return "curl -X POST " + window.location.origin + "/api/predict \\\n" +
      "  -H \"Content-Type: application/json\" \\\n" +
      "  -d '" + JSON.stringify(body) + "'";
  }

  /* ---- wire up everything inside #app (re-run after each swap) ---- */
  function init() {
    var form = document.getElementById("predict-form");
    if (!form) return;
    var data = profiles();

    form.querySelectorAll(".field").forEach(bindField);

    var chips = form.querySelectorAll(".chip[data-crop]");
    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        var profile = data[chip.getAttribute("data-crop")];
        if (!profile) return;
        setValues(form, profile);
        chips.forEach(function (c) { c.setAttribute("aria-pressed", String(c === chip)); });
      });
    });

    var clear = document.getElementById("clear-btn");
    if (clear) clear.addEventListener("click", function () {
      form.querySelectorAll('input[type="number"]').forEach(function (input) {
        input.value = "";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });
      chips.forEach(function (c) { c.setAttribute("aria-pressed", "false"); });
    });

    var copy = document.getElementById("copy-curl");
    if (copy) copy.addEventListener("click", function () {
      var done = function () { copy.textContent = "Copied"; setTimeout(function () { copy.textContent = "Copy as cURL"; }, 1500); };
      if (navigator.clipboard) navigator.clipboard.writeText(curlFor(form)).then(done, function () {});
    });

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      submit(form);
    });
  }

  /* ---- submit without a page reload; fall back to a normal post on any problem ---- */
  function submit(form) {
    var button = form.querySelector('button[type="submit"]');
    button.setAttribute("aria-busy", "true");
    button.disabled = true;

    fetch(form.getAttribute("action") || window.location.pathname, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded", "X-Requested-With": "fetch" },
      body: new URLSearchParams(new FormData(form)).toString()
    }).then(function (response) {
      if (response.status === 429 || response.status >= 500) throw new Error("status " + response.status);
      return response.text();
    }).then(function (html) {
      var next = new DOMParser().parseFromString(html, "text/html").getElementById("app");
      var current = document.getElementById("app");
      if (!next || !current) throw new Error("unexpected response");
      current.replaceWith(next);
      init();
      var target = document.getElementById("result-panel") || document.querySelector(".has-error");
      if (target) target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    }).catch(function () {
      button.removeAttribute("aria-busy");
      button.disabled = false;
      form.submit();  // classic form post as a fallback
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
