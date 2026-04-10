/* ══════════════════════════════════════════════════════
   DiabetesIQ – Frontend Logic
   - Syncs sliders ↔ number inputs
   - Calls /model-info on load for feature importance chart
   - Calls /predict on form submit → animates gauge + result
   ══════════════════════════════════════════════════════ */

"use strict";

// ── Slider ↔ Number Input sync ──────────────────────────
document.querySelectorAll(".slider").forEach(slider => {
  const targetId = slider.dataset.target;
  const numInput = document.getElementById(targetId);

  // slider → number
  slider.addEventListener("input", () => {
    numInput.value = slider.value;
  });
  // number → slider
  numInput.addEventListener("input", () => {
    slider.value = numInput.value;
  });
});

// ── Load model info ──────────────────────────────────────
let importanceChart = null;

async function loadModelInfo() {
  try {
    const res  = await fetch("/model-info");
    const data = await res.json();

    // Header badge
    document.getElementById("model-badge").textContent =
      `RF · Acc ${data.accuracy}% · AUC ${data.auc}`;

    // Footer
    document.getElementById("footer-accuracy").textContent = `${data.accuracy}%`;
    document.getElementById("footer-auc").textContent      = data.auc;

    // Feature importance bar chart
    const labels = Object.keys(data.feature_importances);
    const values = Object.values(data.feature_importances).map(v => +(v * 100).toFixed(2));

    const ctx = document.getElementById("importance-chart").getContext("2d");
    importanceChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data:            values,
          backgroundColor: labels.map((_, i) =>
            `hsla(${200 + i * 25}, 80%, 65%, 0.75)`
          ),
          borderColor: labels.map((_, i) =>
            `hsla(${200 + i * 25}, 80%, 65%, 1)`
          ),
          borderWidth:  1,
          borderRadius: 6,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.parsed.x.toFixed(2)}% importance`,
            },
          },
        },
        scales: {
          x: {
            grid:   { color: "rgba(255,255,255,0.05)" },
            ticks:  { color: "#94a3b8", font: { size: 11 } },
          },
          y: {
            grid:   { display: false },
            ticks:  { color: "#e2e8f0", font: { size: 12, weight: "600" } },
          },
        },
      },
    });
  } catch (e) {
    console.error("Failed to load model info:", e);
  }
}

// ── Gauge drawing ────────────────────────────────────────
function drawGauge(ctx, percent, riskLevel) {
  const W = 220, H = 130;
  const cx = W / 2, cy = H - 10;
  const r  = 90;
  const startAngle = Math.PI;
  const endAngle   = 2 * Math.PI;
  const fillAngle  = startAngle + (percent / 100) * Math.PI;

  const colorMap = {
    Low:      "#34d399",
    Moderate: "#fbbf24",
    High:     "#f87171",
  };
  const color = colorMap[riskLevel] || "#a78bfa";

  ctx.clearRect(0, 0, W, H);

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth   = 18;
  ctx.lineCap     = "round";
  ctx.stroke();

  // Fill with gradient
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0,   "#a78bfa");
  grad.addColorStop(0.5, color);
  grad.addColorStop(1,   color);

  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, fillAngle);
  ctx.strokeStyle = grad;
  ctx.lineWidth   = 18;
  ctx.lineCap     = "round";
  ctx.stroke();

  // Glow effect
  ctx.shadowBlur  = 20;
  ctx.shadowColor = color;
  ctx.beginPath();
  ctx.arc(cx, cy, r, fillAngle - 0.01, fillAngle);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 22;
  ctx.lineCap     = "round";
  ctx.stroke();
  ctx.shadowBlur = 0;
}

// ── Animate gauge counter ────────────────────────────────
function animateGauge(targetPercent, riskLevel) {
  const canvas  = document.getElementById("gauge-canvas");
  const ctx     = canvas.getContext("2d");
  const el      = document.getElementById("gauge-percent");
  const duration = 1000; // ms
  const start    = performance.now();

  function tick(now) {
    const elapsed  = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // ease-out
    const eased    = 1 - Math.pow(1 - progress, 3);
    const current  = Math.round(targetPercent * eased);

    drawGauge(ctx, current, riskLevel);
    el.textContent = `${current}%`;

    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Prediction form submit ───────────────────────────────
const FEATURE_COLS = [
  "Pregnancies","Glucose","BloodPressure",
  "Insulin","BMI","DiabetesPedigreeFunction","Age"
];

document.getElementById("predict-form").addEventListener("submit", async e => {
  e.preventDefault();

  // Gather values
  const body = {};
  FEATURE_COLS.forEach(col => {
    body[col] = parseFloat(document.getElementById(col).value) || 0;
  });

  // UI: loading state
  const btn      = document.getElementById("submit-btn");
  const btnText  = document.getElementById("btn-text");
  const btnLoader= document.getElementById("btn-loader");
  btn.disabled   = true;
  btnText.classList.add("hidden");
  btnLoader.classList.remove("hidden");

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    // Show result section
    document.getElementById("result-placeholder").classList.add("hidden");
    const content = document.getElementById("result-content");
    content.classList.remove("hidden");
    content.style.animation = "none";
    // force reflow
    void content.offsetHeight;
    content.style.animation = "fadeUp .5s ease both";

    // Gauge
    animateGauge(data.probability, data.risk_level);

    // Risk badge
    const badge = document.getElementById("risk-badge");
    badge.textContent = `${data.risk_level} Risk`;
    badge.className   = `risk-badge risk-${data.risk_level}`;

    // Verdict
    const verdictEl = document.getElementById("result-verdict");
    if (data.prediction === 1) {
      verdictEl.innerHTML = `
        <span style="color:#f87171">⚠️ Diabetes risk detected.</span><br>
        <span style="color:#94a3b8;font-size:0.85rem;">Confidence: ${data.probability}%</span>`;
    } else {
      verdictEl.innerHTML = `
        <span style="color:#34d399">✅ Low likelihood of diabetes.</span><br>
        <span style="color:#94a3b8;font-size:0.85rem;">Confidence: ${(100 - data.probability).toFixed(1)}% negative</span>`;
    }

    // Advice
    const adviceEl = document.getElementById("advice-list");
    adviceEl.innerHTML = "";
    if (data.advice && data.advice.length > 0) {
      data.advice.forEach(tip => {
        const li = document.createElement("li");
        li.textContent = tip;
        adviceEl.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "Your biomarkers appear within normal ranges. Maintain a healthy lifestyle!";
      adviceEl.appendChild(li);
    }

    // Scroll result into view
    document.getElementById("result-card").scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    btn.disabled = false;
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
  }
});

// ── Init ─────────────────────────────────────────────────
loadModelInfo();
