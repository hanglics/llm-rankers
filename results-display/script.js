const docCatalog = {
  A: "Broad overview passage",
  B: "Exact entity match",
  C: "Supporting evidence",
  D: "Weak partial match",
  E: "High-value nuance",
  F: "Clear distractor",
};

const algorithms = [
  {
    id: "topdown",
    kicker: "Best-only baseline",
    title: "TopDown",
    badge: "stable baseline",
    summary:
      "Ask only for the best document. The signal stays clean, but every call only places one item.",
    bullets: [
      "Strongest efficiency baseline in practical end-to-end terms",
      "Keeps the comparison task simple",
      "Leaves the worst-side signal completely unused",
    ],
    steps: [
      {
        narrative:
          "The model sees the full candidate window and returns only the strongest item. One document is placed; the rest stay unresolved.",
        top: [],
        active: ["A", "B", "C", "D", "E", "F"],
        bottom: [],
        best: "B",
      },
      {
        narrative:
          "After the first winner is removed, the next pass looks for the next best item. Progress is steady, but still one-sided.",
        top: ["B"],
        active: ["A", "C", "D", "E", "F"],
        bottom: [],
        best: "E",
      },
      {
        narrative:
          "The ranking fills from the top downward. Useful, reliable, and relatively cheap.",
        top: ["B", "E"],
        active: ["A", "C", "D", "F"],
        bottom: [],
        best: "C",
      },
      {
        narrative:
          "TopDown ends with a strong front of the ranking, but the bottom side of the list was never explicitly modeled.",
        top: ["B", "E", "C"],
        active: ["A", "D", "F"],
        bottom: [],
        best: "A",
      },
    ],
  },
  {
    id: "bottomup",
    kicker: "Worst-only probe",
    title: "BottomUp",
    badge: "diagnostic failure",
    summary:
      "Ask only for the least relevant document. In theory this should be symmetric; in practice it becomes noisy and recency-biased.",
    bullets: [
      "Loses in all 18 TREC DL configurations",
      "Has 6 Bonferroni-significant losses versus best TopDown",
      "Useful mainly because it reveals directional asymmetry",
    ],
    steps: [
      {
        narrative:
          "The prompt asks for the worst document first. That sounds elegant, but the signal is much less stable than best-only selection.",
        top: [],
        active: ["A", "B", "C", "D", "E", "F"],
        bottom: [],
        worst: "F",
      },
      {
        narrative:
          "The method keeps stripping items from the bottom. This often amplifies position bias rather than clarifying true irrelevance.",
        top: [],
        active: ["A", "B", "C", "D", "E"],
        bottom: ["F"],
        worst: "D",
      },
      {
        narrative:
          "Even when some weak passages are correctly removed, the remaining order is much noisier than the TopDown order.",
        top: [],
        active: ["A", "B", "C", "E"],
        bottom: ["D", "F"],
        worst: "A",
      },
      {
        narrative:
          "BottomUp exposes a useful phenomenon for the paper, but not a replacement algorithm you would actually want to deploy.",
        top: [],
        active: ["B", "C", "E"],
        bottom: ["A", "D", "F"],
        worst: "C",
      },
    ],
  },
  {
    id: "dualend",
    kicker: "Joint elicitation",
    title: "DualEnd",
    badge: "best family overall",
    summary:
      "Ask for best and worst in the same prompt. This is the only bidirectional idea that reliably helps, even if the gains are still modest and expensive.",
    bullets: [
      "Positive in 14 of 18 TREC DL configurations",
      "Wins all 12 Qwen configurations",
      "Best current interpretation: same-call worst acts as a weak auxiliary signal",
    ],
    steps: [
      {
        narrative:
          "One prompt returns both ends of the candidate window. The top and bottom of the list begin to settle at the same time.",
        top: [],
        active: ["A", "B", "C", "D", "E", "F"],
        bottom: [],
        best: "B",
        worst: "F",
      },
      {
        narrative:
          "Because both edges move together, the active uncertainty zone shrinks faster than in the one-sided baselines.",
        top: ["B"],
        active: ["A", "C", "D", "E"],
        bottom: ["F"],
        best: "E",
        worst: "D",
      },
      {
        narrative:
          "This is where the method gets its lift: it mostly preserves the TopDown front while adding a same-call constraint from the bottom side.",
        top: ["B", "E"],
        active: ["A", "C"],
        bottom: ["D", "F"],
        best: "C",
        worst: "A",
      },
      {
        narrative:
          "DualEnd is currently the best family overall, but the cost is high enough that the next step should be a selective or bias-aware variant.",
        top: ["B", "E", "C"],
        active: [],
        bottom: ["A", "D", "F"],
        best: null,
        worst: null,
      },
    ],
  },
  {
    id: "bidir",
    kicker: "Independent fusion",
    title: "BiDir",
    badge: "imports noise",
    summary:
      "Run TopDown and BottomUp independently, then fuse the rankings. The idea sounds robust, but the BottomUp channel is too weak to rescue.",
    bullets: [
      "Average delta versus best TopDown is negative",
      "Has 3 Bonferroni-significant losses",
      "Better lesson: same-call signals matter more than independent fusion",
    ],
    layout: "split",
    steps: [
      {
        narrative:
          "BiDir starts by paying for two ranking processes: a reliable TopDown pass and a much weaker BottomUp pass.",
        leftTop: ["B"],
        leftActive: ["A", "C", "D", "E", "F"],
        rightActive: ["A", "B", "C", "D", "E"],
        rightBottom: ["F"],
        fusion: "Collect two rank lists",
      },
      {
        narrative:
          "The fusion stage receives one clean signal and one noisy one. In practice, the second channel rarely adds complementary evidence.",
        leftTop: ["B", "E"],
        leftActive: ["A", "C", "D", "F"],
        rightActive: ["A", "B", "C", "E"],
        rightBottom: ["D", "F"],
        fusion: "RRF / weighted fuse",
      },
      {
        narrative:
          "The fused output usually lands between the stable baseline and the noisy bottom-up channel, which is why it seldom beats pure TopDown.",
        leftTop: ["B", "E", "C"],
        leftActive: ["A", "D", "F"],
        rightActive: ["B", "C", "E"],
        rightBottom: ["A", "D", "F"],
        fusion: "Small gain, often negative",
      },
    ],
  },
];

let currentAlgorithmIndex = 0;
let currentStepIndex = 0;
let autoplay = true;
let timerId = null;

const tabsEl = document.getElementById("algoTabs");
const stageEl = document.getElementById("algorithmStage");
const algoKickerEl = document.getElementById("algoKicker");
const algoTitleEl = document.getElementById("algoTitle");
const algoSummaryEl = document.getElementById("algoSummary");
const algoStepIndexEl = document.getElementById("algoStepIndex");
const algoBadgeEl = document.getElementById("algoBadge");
const stepNarrativeEl = document.getElementById("stepNarrative");
const algoBulletsEl = document.getElementById("algoBullets");
const playToggleEl = document.getElementById("playToggle");
const prevStepEl = document.getElementById("prevStep");
const nextStepEl = document.getElementById("nextStep");

function buildTabs() {
  algorithms.forEach((algo, index) => {
    const button = document.createElement("button");
    button.className = "algo-tab";
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", index === currentAlgorithmIndex ? "true" : "false");
    button.textContent = algo.title;
    button.addEventListener("click", () => {
      currentAlgorithmIndex = index;
      currentStepIndex = 0;
      render();
      restartAutoplay();
    });
    tabsEl.appendChild(button);
  });
}

function createDocCard(label, { active = false, best = false, worst = false } = {}) {
  const card = document.createElement("div");
  card.className = "doc-card";
  if (active) card.classList.add("is-active");
  if (best) card.classList.add("is-best");
  if (worst) card.classList.add("is-worst");

  const labelEl = document.createElement("div");
  labelEl.className = "label";
  labelEl.textContent = label;

  const textEl = document.createElement("div");
  textEl.className = "text";
  textEl.textContent = docCatalog[label];

  card.append(labelEl, textEl);

  if (best) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = "best";
    card.appendChild(badge);
  }

  if (worst) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = "worst";
    card.appendChild(badge);
  }

  return card;
}

function createLane(title, docs, step) {
  const lane = document.createElement("div");
  lane.className = "stage-lane";

  const heading = document.createElement("h4");
  heading.textContent = title;

  const row = document.createElement("div");
  row.className = "card-row";

  docs.forEach((label) => {
    row.appendChild(
      createDocCard(label, {
        active: title === "Active Window",
        best: step.best === label,
        worst: step.worst === label,
      }),
    );
  });

  lane.append(heading, row);
  return lane;
}

function renderStandardStage(algo, step) {
  const grid = document.createElement("div");
  grid.className = "stage-grid";

  grid.appendChild(createLane("Resolved Top", step.top, step));
  grid.appendChild(createLane("Active Window", step.active, step));
  grid.appendChild(createLane("Resolved Bottom", step.bottom, step));

  const caption = document.createElement("div");
  caption.className = "stage-caption";
  caption.innerHTML =
    "<strong>Visual cue:</strong> teal cards are current best picks, coral cards are current worst picks, and lavender outlines mark the actively compared window.";

  grid.appendChild(caption);
  return grid;
}

function renderBidirColumn(title, topDocs, activeDocs, bottomDocs, activeLabel) {
  const column = document.createElement("div");
  column.className = "bidir-column";

  const titleEl = document.createElement("div");
  titleEl.className = "stage-caption";
  titleEl.innerHTML = `<strong>${title}</strong><br>${activeLabel}`;

  column.appendChild(titleEl);
  column.appendChild(
    createLane(title === "TopDown channel" ? "Resolved Top" : "Active Window", topDocs, {
      best: null,
      worst: null,
    }),
  );

  if (activeDocs.length) {
    column.appendChild(
      createLane("Active Window", activeDocs, {
        best: null,
        worst: null,
      }),
    );
  }

  column.appendChild(
    createLane(title === "TopDown channel" ? "Pending Tail" : "Resolved Bottom", bottomDocs, {
      best: null,
      worst: null,
    }),
  );

  return column;
}

function renderSplitStage(algo, step) {
  const wrap = document.createElement("div");
  wrap.className = "bidir-layout";

  wrap.appendChild(
    renderBidirColumn(
      "TopDown channel",
      step.leftTop,
      step.leftActive,
      [],
      "clean best-only ranking path",
    ),
  );

  const fusion = document.createElement("div");
  fusion.className = "fusion-node";
  fusion.innerHTML = `
    <div class="fusion-ring">
      <div class="fusion-core">Fuse</div>
    </div>
    <div class="fusion-note">${step.fusion}</div>
  `;

  wrap.appendChild(fusion);

  wrap.appendChild(
    renderBidirColumn(
      "BottomUp channel",
      [],
      step.rightActive,
      step.rightBottom,
      "noisier worst-only ranking path",
    ),
  );

  return wrap;
}

function updateSidePanel(algo, step) {
  algoKickerEl.textContent = algo.kicker;
  algoTitleEl.textContent = algo.title;
  algoSummaryEl.textContent = algo.summary;
  algoBadgeEl.textContent = algo.badge;
  algoStepIndexEl.textContent = `Step ${currentStepIndex + 1} / ${algo.steps.length}`;
  stepNarrativeEl.textContent = step.narrative;

  algoBulletsEl.innerHTML = "";
  algo.bullets.forEach((bullet) => {
    const li = document.createElement("li");
    li.textContent = bullet;
    algoBulletsEl.appendChild(li);
  });
}

function render() {
  const algo = algorithms[currentAlgorithmIndex];
  const step = algo.steps[currentStepIndex];

  Array.from(tabsEl.children).forEach((tab, index) => {
    tab.setAttribute("aria-selected", index === currentAlgorithmIndex ? "true" : "false");
  });

  stageEl.innerHTML = "";
  stageEl.appendChild(
    algo.layout === "split" ? renderSplitStage(algo, step) : renderStandardStage(algo, step),
  );
  updateSidePanel(algo, step);
}

function nextStep() {
  const algo = algorithms[currentAlgorithmIndex];
  currentStepIndex = (currentStepIndex + 1) % algo.steps.length;
  render();
}

function prevStep() {
  const algo = algorithms[currentAlgorithmIndex];
  currentStepIndex = (currentStepIndex - 1 + algo.steps.length) % algo.steps.length;
  render();
}

function restartAutoplay() {
  if (timerId) window.clearInterval(timerId);
  if (!autoplay) return;
  timerId = window.setInterval(nextStep, 2400);
}

function setupControls() {
  playToggleEl.addEventListener("click", () => {
    autoplay = !autoplay;
    playToggleEl.textContent = autoplay ? "Pause" : "Play";
    restartAutoplay();
  });

  prevStepEl.addEventListener("click", () => {
    prevStep();
    restartAutoplay();
  });

  nextStepEl.addEventListener("click", () => {
    nextStep();
    restartAutoplay();
  });
}

function setupReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("in-view");
      });
    },
    { threshold: 0.15 },
  );

  document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));
}

function setupCounters() {
  const counters = document.querySelectorAll(".count-up");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const target = Number(el.dataset.target || 0);
        const duration = 1100;
        const start = performance.now();

        function tick(now) {
          const progress = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          el.textContent = Math.round(target * eased);
          if (progress < 1) requestAnimationFrame(tick);
        }

        requestAnimationFrame(tick);
        observer.unobserve(el);
      });
    },
    { threshold: 0.45 },
  );

  counters.forEach((counter) => observer.observe(counter));
}

buildTabs();
render();
setupControls();
setupReveal();
setupCounters();
restartAutoplay();
