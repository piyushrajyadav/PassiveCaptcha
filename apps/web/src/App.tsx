import { useEffect, useRef, useState } from "react";
import { extractFeatures } from "./lib/features";
import { scoreSession } from "./lib/inference";
import { SignalCollector } from "./lib/signalCollector";
import { FeatureVector, ModelScores as TModelScores, SessionAssessment } from "./types";
import { ChallengeOverlay } from "./Challenge";
import { AnalyticsDashboard } from "./Dashboard";

// ── Helper components ────────────────────────────────────────────────────────

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/40 p-6 backdrop-blur">
      <p className="mb-4 text-sm uppercase tracking-[0.35em] text-slate-400">{title}</p>
      {children}
    </section>
  );
}

function MetricRow({
  label,
  value,
  flag,
}: {
  label: string;
  value: string;
  flag?: boolean;
}) {
  return (
    <div className="flex items-center justify-between text-sm">
      <dt className={flag ? "text-rose-300" : "text-slate-300"}>{label}</dt>
      <dd className={`font-mono ${flag ? "text-rose-300" : "text-slate-100"}`}>{value}</dd>
    </div>
  );
}

function SignalBadge({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200">
      {label}
    </span>
  );
}

function ProgressBar({
  value,
  max = 1,
  danger = false,
}: {
  value: number;
  max?: number;
  danger?: boolean;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="h-1.5 w-full rounded-full bg-white/10">
      <div
        className={`h-1.5 rounded-full transition-all duration-500 ${
          danger ? "bg-rose-400" : "bg-sky-400"
        }`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

// ── Phase 2: SHAP bar ─────────────────────────────────────────────────────────

function ShapBar({
  name,
  value,
  maxAbsValue,
}: {
  name: string;
  value: number;
  maxAbsValue: number;
}) {
  const isPositive = value >= 0;
  const pct = maxAbsValue > 0 ? (Math.abs(value) / maxAbsValue) * 100 : 0;
  const label = name
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (s) => s.toUpperCase())
    .trim();

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-3">
        <span className="min-w-0 truncate text-xs text-slate-300">{label}</span>
        <span
          className={`shrink-0 font-mono text-xs ${
            isPositive ? "text-emerald-400" : "text-rose-400"
          }`}
        >
          {value > 0 ? "+" : ""}
          {value.toFixed(4)}
        </span>
      </div>
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        {isPositive ? (
          <div
            className="absolute left-0 top-0 h-full rounded-full bg-emerald-400 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div
            className="absolute right-0 top-0 h-full rounded-full bg-rose-400 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
    </div>
  );
}

// ── Phase 2: model vote card ──────────────────────────────────────────────────

function ModelVoteCard({
  label,
  weight,
  score,
  accent,
}: {
  label: string;
  weight: number;
  score: number | null | undefined;
  accent: "sky" | "violet" | "amber";
}) {
  const colors = {
    sky: {
      ring: "border-sky-500/30",
      bar: "bg-sky-400",
      tag: "text-sky-300",
    },
    violet: {
      ring: "border-violet-500/30",
      bar: "bg-violet-400",
      tag: "text-violet-300",
    },
    amber: {
      ring: "border-amber-500/30",
      bar: "bg-amber-400",
      tag: "text-amber-300",
    },
  } as const;
  const c = colors[accent];
  const available = score !== null && score !== undefined;
  const scoreColor = !available
    ? "text-slate-500"
    : score >= 0.75
    ? "text-emerald-300"
    : score <= 0.45
    ? "text-rose-300"
    : "text-amber-200";

  return (
    <div className={`rounded-xl border ${c.ring} bg-white/5 p-4`}>
      <div className="flex items-start justify-between">
        <p className="text-xs text-slate-400">{label}</p>
        <span className={`rounded-full border ${c.ring} px-2 py-0.5 text-xs ${c.tag}`}>
          {weight}%
        </span>
      </div>
      <p className={`mt-1 text-2xl font-semibold tabular-nums ${scoreColor}`}>
        {available ? (score as number).toFixed(3) : "—"}
      </p>
      <div className="mt-3">
        <div className="h-1.5 w-full rounded-full bg-white/10">
          <div
            className={`h-1.5 rounded-full transition-all duration-700 ${
              available ? c.bar : "bg-white/5"
            }`}
            style={{ width: available ? `${(score as number) * 100}%` : "0%" }}
          />
        </div>
        {available ? (
          <p className={`mt-1.5 text-xs ${scoreColor}`}>
            {(score as number) >= 0.75
              ? "human"
              : (score as number) <= 0.45
              ? "bot"
              : "review"}
          </p>
        ) : (
          <p className="mt-1.5 text-xs text-slate-500">not trained yet — run training scripts</p>
        )}
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

export default function App() {
  const collectorRef = useRef<SignalCollector | null>(null);
  const [assessment, setAssessment] = useState<SessionAssessment | null>(null);
  const [features, setFeatures] = useState<FeatureVector | null>(null);
  const apiBaseUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
  const sessionTokenRef = useRef<string | null>(null);
  const [challengeSolved, setChallengeSolved] = useState(false);

  useEffect(() => {
    // Phase 5 Enterprise Flow: Init session and get JWT
    const initSession = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/v1/init`, { method: "POST" });
        const data = await res.json();
        sessionTokenRef.current = data.token;
      } catch (e) {
        console.error("Failed to init token", e);
      }
    };
    initSession();

    const collector = new SignalCollector();
    collectorRef.current = collector;
    const cleanup = collector.start();

    const interval = window.setInterval(async () => {
      const snapshot = collector.snapshot();
      const extracted = extractFeatures(snapshot);
      setFeatures(extracted);

      try {
        const headers: Record<string, string> = { "Content-Type": "application/json" };
        if (sessionTokenRef.current) {
          headers["Authorization"] = `Bearer ${sessionTokenRef.current}`;
        }

        const response = await fetch(`${apiBaseUrl}/v1/ingest`, {
          method: "POST",
          headers,
          body: JSON.stringify(snapshot),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = (await response.json()) as SessionAssessment;
        
        // Prevent backend from overriding a successful manual challenge with a new "review"
        setAssessment((prev) => {
          if (challengeSolved && result.verdict === "review") {
             return { ...result, verdict: "human", score: Math.max(result.score, 0.8) };
          }
          return result;
        });

      } catch {
        setAssessment(scoreSession(extracted));
      }
    }, 1000);

    return () => {
      cleanup();
      window.clearInterval(interval);
    };
  }, [apiBaseUrl, challengeSolved]);

  const verdict = assessment?.verdict ?? "review";
  const score = assessment?.score ?? 0.5;
  const flagged = assessment?.flaggedSignals ?? [];
  const f = features;

  // Phase 2 — ensemble + SHAP
  const modelScores: TModelScores | undefined = assessment?.modelScores;
  const shapValues: Record<string, number> = assessment?.shapValues ?? {};
  const shapEntries = Object.entries(shapValues)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    .slice(0, 10);
  const maxShap =
    shapEntries.length > 0
      ? Math.max(...shapEntries.map(([, v]) => Math.abs(v)))
      : 1;

  const verdictColors = {
    human: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    bot: "bg-rose-500/20 text-rose-300 border-rose-500/30",
    review: "bg-amber-500/20 text-amber-200 border-amber-500/30",
  };

  const scoreColor =
    verdict === "human"
      ? "text-emerald-300"
      : verdict === "bot"
      ? "text-rose-300"
      : "text-amber-200";

  return (
    <>
      <main className="min-h-screen bg-[radial-gradient(circle_at_top,#1f3b73,transparent_45%),linear-gradient(135deg,#07111f,#10223d_45%,#091726)] px-6 py-10 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-8">

        {/* ── Header ──────────────────────────────────────────────────── */}
        <header className="rounded-[2rem] border border-white/10 bg-white/5 px-8 py-7 shadow-2xl shadow-sky-950/30 backdrop-blur">
          <p className="text-sm uppercase tracking-[0.4em] text-sky-200/80">PassiveCaptcha</p>
          <h1 className="mt-3 font-serif text-4xl leading-tight text-white">
            Invisible bot scoring — behavioral evidence, zero CAPTCHA friction.
          </h1>
          <p className="mt-3 max-w-3xl text-base text-slate-300">
            Every mouse move, keystroke, scroll event, and browser fingerprint signal is analysed
            passively by a three-model ensemble. The panels below update live as you interact.
          </p>
        </header>

        {/* ── Score + Demo form ────────────────────────────────────────── */}
        <div className="grid gap-6 lg:grid-cols-[1fr_380px]">

          {/* Demo login form */}
          <Card title="Demo interaction surface">
            <p className="mb-5 text-sm text-slate-400">
              Interact naturally — move your mouse, type, scroll. The engine scores your session
              passively. No puzzles, no clicks required.
            </p>
            <form className="grid gap-4">
              <div>
                <label className="mb-2 block text-sm text-slate-300" htmlFor="email">
                  Work email
                </label>
                <input
                  id="email"
                  type="email"
                  placeholder="analyst@company.com"
                  className="w-full rounded-xl border border-white/10 bg-white/10 px-4 py-3 text-white placeholder:text-slate-400 outline-none"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm text-slate-300" htmlFor="password">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  className="w-full rounded-xl border border-white/10 bg-white/10 px-4 py-3 text-white placeholder:text-slate-400 outline-none"
                />
              </div>
              <button
                type="button"
                className="rounded-xl bg-sky-300 px-4 py-3 font-semibold text-slate-950 transition hover:bg-sky-200"
              >
                Continue
              </button>
            </form>
          </Card>

          {/* Live verdict */}
          <div className="space-y-4">
            <Card title="Live verdict">
              <div className="flex items-end justify-between">
                <div>
                  <p className={`text-7xl font-semibold tabular-nums ${scoreColor}`}>
                    {score.toFixed(2)}
                  </p>
                  <p className="mt-2 text-xs text-slate-500">
                    1 = definitely human · 0 = definitely bot
                  </p>
                </div>
                <span
                  className={`rounded-full border px-4 py-2 text-sm font-semibold ${verdictColors[verdict]}`}
                >
                  {verdict.toUpperCase()}
                </span>
              </div>
              <div className="mt-4">
                <ProgressBar value={score} max={1} danger={verdict === "bot"} />
              </div>
              <p className="mt-3 text-xs text-slate-500">
                Ensemble ML scorer (RF + XGBoost + LSTM) when reachable · local heuristic fallback
                otherwise
              </p>
            </Card>

            <Card title="Flagged signals">
              <div className="flex flex-wrap gap-2">
                {flagged.length > 0 ? (
                  flagged.map((s) => <SignalBadge key={s} label={s} />)
                ) : (
                  <span className="text-sm text-slate-500">none — session looks clean</span>
                )}
              </div>
            </Card>
          </div>
        </div>

        {/* ── Phase 2: Ensemble model breakdown ───────────────────────── */}
        <Card title="Ensemble · three-model breakdown">
          <p className="mb-5 text-sm text-slate-400">
            Three independent classifiers vote on P(human). Their probabilities are combined with
            fixed weights{" "}
            <span className="text-slate-300">RF 25% · XGBoost 45% · LSTM 30%</span>. When a model
            has not been trained yet its weight is redistributed to the others.
          </p>

          {/* Model vote cards */}
          <div className="grid gap-4 sm:grid-cols-3">
            <ModelVoteCard
              label="RandomForest"
              weight={25}
              score={modelScores?.randomForest}
              accent="sky"
            />
            <ModelVoteCard
              label="XGBoost (calibrated)"
              weight={45}
              score={modelScores?.xgboost}
              accent="violet"
            />
            <ModelVoteCard
              label="LSTM (mouse sequence)"
              weight={30}
              score={modelScores?.lstm}
              accent="amber"
            />
          </div>

          {/* Ensemble summary bar */}
          <div className="mt-5 rounded-xl border border-white/10 bg-white/5 p-4">
            <div className="flex items-center justify-between">
              <p className="text-sm text-slate-300">Ensemble score</p>
              <p className={`text-2xl font-semibold tabular-nums ${scoreColor}`}>
                {modelScores ? modelScores.ensemble.toFixed(3) : score.toFixed(3)}
              </p>
            </div>
            <div className="mt-3">
              <ProgressBar
                value={modelScores?.ensemble ?? score}
                max={1}
                danger={verdict === "bot"}
              />
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Weighted average of all available model scores. Decision thresholds:
              human ≥ 0.75 · review 0.45 – 0.75 · bot ≤ 0.45
            </p>
          </div>
        </Card>

        {/* ── Phase 2: SHAP explainability ────────────────────────────── */}
        <Card title="SHAP · feature contributions">
          <p className="mb-5 text-sm text-slate-400">
            SHAP (SHapley Additive exPlanations) shows which features pushed this session&apos;s
            score toward{" "}
            <span className="text-emerald-400">human ↑</span> or{" "}
            <span className="text-rose-400">bot ↓</span>. Values come from the XGBoost base
            estimator (falls back to RandomForest). Only available when the backend is reachable
            and at least one model has been trained.
          </p>

          {shapEntries.length > 0 ? (
            <div className="space-y-3">
              {shapEntries.map(([name, value]) => (
                <ShapBar key={name} name={name} value={value} maxAbsValue={maxShap} />
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-white/5 bg-white/5 p-6 text-center">
              <p className="text-sm text-slate-500">
                No SHAP values yet.
              </p>
              <p className="mt-1 text-xs text-slate-600">
                Train the models and connect to the FastAPI backend to see per-feature
                explanations.
              </p>
            </div>
          )}

          {shapEntries.length > 0 && (
            <div className="mt-4 flex gap-4 text-xs text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2 w-4 rounded-full bg-emerald-400" />
                pushes toward human
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2 w-4 rounded-full bg-rose-400" />
                pushes toward bot
              </span>
            </div>
          )}
        </Card>

        {/* ── Phase 1: Feature panels grid ─────────────────────────────── */}
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">

          {/* Mouse dynamics */}
          <Card title="Mouse dynamics">
            <dl className="space-y-3">
              <MetricRow
                label="Speed mean"
                value={f ? f.mouseSpeedMean.toFixed(3) : "—"}
              />
              <MetricRow
                label="Speed std dev"
                value={f ? f.mouseSpeedStd.toFixed(3) : "—"}
                flag={!!f && f.mouseSpeedStd < 0.05 && f.sampleCount > 20}
              />
              <MetricRow
                label="Linearity"
                value={f ? f.mouseLinearity.toFixed(3) : "—"}
                flag={!!f && f.mouseLinearity > 0.98}
              />
              <MetricRow
                label="Acceleration mean"
                value={f ? f.mouseAccelerationMean.toFixed(4) : "—"}
                flag={!!f && f.mouseAccelerationMean < 0.01 && f.sampleCount > 20}
              />
              <MetricRow
                label="Jerk mean"
                value={f ? f.mouseJerkMean.toFixed(5) : "—"}
                flag={!!f && f.mouseJerkMean < 0.005 && f.sampleCount > 20}
              />
              <MetricRow
                label="Click count"
                value={f ? String(f.clickCount) : "—"}
              />
              <div className="pt-1">
                <p className="mb-1 text-xs text-slate-500">Linearity</p>
                <ProgressBar
                  value={f ? f.mouseLinearity : 0}
                  max={1}
                  danger={!!f && f.mouseLinearity > 0.98}
                />
              </div>
            </dl>
          </Card>

          {/* Scroll behaviour */}
          <Card title="Scroll behaviour">
            <dl className="space-y-3">
              <MetricRow
                label="Scroll events"
                value={f ? String(f.scrollEvents) : "—"}
              />
              <MetricRow
                label="Velocity mean"
                value={f ? f.scrollVelocityMean.toFixed(3) : "—"}
              />
              <MetricRow
                label="Velocity std dev"
                value={f ? f.scrollVelocityStd.toFixed(3) : "—"}
                flag={!!f && f.scrollEvents > 5 && f.scrollVelocityStd < 0.01}
              />
              <MetricRow
                label="Direction changes"
                value={f ? String(f.scrollDirectionChanges) : "—"}
              />
            </dl>
            <div className="mt-4 rounded-xl border border-white/5 bg-white/5 p-3 text-xs text-slate-400">
              <span className="text-slate-300">Bot pattern: </span>
              constant velocity, zero direction reversals
            </div>
          </Card>

          {/* Keyboard cadence */}
          <Card title="Keyboard cadence">
            <dl className="space-y-3">
              <MetricRow
                label="Key events"
                value={f ? String(f.keyEvents) : "—"}
              />
              <MetricRow
                label="Interval mean"
                value={f ? `${f.averageKeyInterval.toFixed(0)} ms` : "—"}
                flag={!!f && f.averageKeyInterval > 0 && f.averageKeyInterval < 20}
              />
              <MetricRow
                label="Interval std dev"
                value={f ? `${f.keyIntervalStd.toFixed(1)} ms` : "—"}
                flag={!!f && f.keyEvents > 4 && f.keyIntervalStd < 5}
              />
              <MetricRow
                label="Hold duration mean"
                value={f ? `${f.keyHoldDurationMean.toFixed(1)} ms` : "—"}
                flag={!!f && f.keyHoldDurationMean > 0 && f.keyHoldDurationMean < 15}
              />
            </dl>
            <div className="mt-4 rounded-xl border border-white/5 bg-white/5 p-3 text-xs text-slate-400">
              <span className="text-slate-300">Bot pattern: </span>
              interval &lt; 20 ms, std ≈ 0, hold ≈ 0 ms
            </div>
          </Card>

          {/* Timing & visibility */}
          <Card title="Timing & visibility">
            <dl className="space-y-3">
              <MetricRow
                label="First interaction"
                value={
                  f && f.timeToFirstInteractionMs !== null
                    ? `${f.timeToFirstInteractionMs} ms`
                    : "—"
                }
                flag={
                  !!f &&
                  f.timeToFirstInteractionMs !== null &&
                  f.timeToFirstInteractionMs < 300
                }
              />
              <MetricRow
                label="Visibility changes"
                value={f ? String(f.visibilityChanges) : "—"}
                flag={!!f && f.visibilityChanges > 3}
              />
              <MetricRow
                label="Change rate"
                value={f ? `${f.visibilityChangeRate.toFixed(2)}/min` : "—"}
              />
              <MetricRow
                label="Mean gap between changes"
                value={
                  f && f.meanTimeBetweenVisibilityChanges > 0
                    ? `${f.meanTimeBetweenVisibilityChanges.toFixed(0)} ms`
                    : "—"
                }
                flag={
                  !!f &&
                  f.visibilityChanges > 1 &&
                  f.meanTimeBetweenVisibilityChanges > 0 &&
                  f.meanTimeBetweenVisibilityChanges < 200
                }
              />
              <MetricRow
                label="Sample count"
                value={f ? String(f.sampleCount) : "—"}
              />
            </dl>
          </Card>
        </div>

        {/* ── Phase 1: Browser entropy panel ───────────────────────────── */}
        <Card title="Browser entropy & fingerprint signals">
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-5">

            {/* WebGL */}
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-400">WebGL renderer</p>
              <p
                className={`mt-2 text-lg font-semibold ${
                  f && f.hasWebGL === 0 ? "text-rose-300" : "text-emerald-300"
                }`}
              >
                {f ? (f.hasWebGL === 1 ? "Present ✓" : "Missing ✗") : "—"}
              </p>
              <p className="mt-1 text-xs text-slate-500">Absent in most headless browsers</p>
            </div>

            {/* Viewport ratio */}
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-400">Viewport / screen ratio</p>
              <p
                className={`mt-2 text-lg font-semibold ${
                  f && f.screenViewportRatio > 0.99 ? "text-rose-300" : "text-emerald-300"
                }`}
              >
                {f ? f.screenViewportRatio.toFixed(3) : "—"}
              </p>
              <p className="mt-1 text-xs text-slate-500">Headless ≈ 1.000 exactly</p>
            </div>

            {/* Device pixel ratio */}
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-400">Device pixel ratio</p>
              <p
                className={`mt-2 text-lg font-semibold ${
                  f && f.devicePixelRatio === 1 && f.hasWebGL === 0
                    ? "text-rose-300"
                    : "text-emerald-300"
                }`}
              >
                {f ? f.devicePixelRatio.toFixed(2) : "—"}
              </p>
              <p className="mt-1 text-xs text-slate-500">Real HiDPI devices &gt; 1</p>
            </div>

            {/* Touch points */}
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-400">Max touch points</p>
              <p className="mt-2 text-lg font-semibold text-slate-100">
                {f ? f.touchPoints : "—"}
              </p>
              <p className="mt-1 text-xs text-slate-500">0 on desktop, &gt;0 on mobile</p>
            </div>

            {/* Suspicious UA */}
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-400">User-agent risk</p>
              <p
                className={`mt-2 text-lg font-semibold ${
                  f && f.isSuspiciousUA === 1 ? "text-rose-300" : "text-emerald-300"
                }`}
              >
                {f ? (f.isSuspiciousUA === 1 ? "Suspicious ✗" : "Clean ✓") : "—"}
              </p>
              <p className="mt-1 text-xs text-slate-500">Headless / automation strings</p>
            </div>
          </div>
        </Card>

          {/* ── Phase 4: Analytics Dashboard ─────────────────────────────────────────────────── */}
          <AnalyticsDashboard assessment={assessment} features={features} />

          {/* ── Footer ───────────────────────────────────────────────────────────── */}
          <footer className="pb-4 mt-8 text-center text-xs text-slate-600">
            PassiveCaptcha · all signals collected passively · no cookies · no CAPTCHA
          </footer>

        </div>
      </main>

      {/* ── Phase 3: Challenge Overlay ───────────────────────────────────────────────────── */}
      {!challengeSolved && verdict === "review" && (
        <ChallengeOverlay onComplete={(success) => {
          if (success && assessment) {
            setChallengeSolved(true);
            setAssessment({ ...assessment, verdict: "human", score: Math.max(assessment.score, 0.8) });
          }
        }} />
      )}
    </>
  );
}
