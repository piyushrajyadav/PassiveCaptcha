import { useEffect, useRef, useState } from "react";
import { extractFeatures } from "./lib/features";
import { scoreSession } from "./lib/inference";
import { SignalCollector } from "./lib/signalCollector";
import { SessionAssessment } from "./types";

const featureLabels = [
  "Mouse speed and linearity",
  "Click timing and density",
  "Scroll rhythm",
  "Key cadence irregularity",
  "Visibility changes",
  "Time to first interaction"
];

export default function App() {
  const collectorRef = useRef<SignalCollector | null>(null);
  const [assessment, setAssessment] = useState<SessionAssessment | null>(null);
  const apiBaseUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

  useEffect(() => {
    const collector = new SignalCollector();
    collectorRef.current = collector;
    const cleanup = collector.start();

    const interval = window.setInterval(async () => {
      const snapshot = collector.snapshot();
      const features = extractFeatures(snapshot);

      try {
        const response = await fetch(`${apiBaseUrl}/v1/ingest`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(snapshot)
        });

        if (!response.ok) {
          throw new Error(`Inference request failed with status ${response.status}`);
        }

        const result = (await response.json()) as SessionAssessment;
        setAssessment(result);
      } catch (_error) {
        setAssessment(scoreSession(features));
      }
    }, 1000);

    return () => {
      cleanup();
      window.clearInterval(interval);
    };
  }, [apiBaseUrl]);

  const verdictTone =
    assessment?.verdict === "human"
      ? "bg-emerald-500/20 text-emerald-200"
      : assessment?.verdict === "bot"
        ? "bg-rose-500/20 text-rose-200"
        : "bg-amber-500/20 text-amber-100";

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,#1f3b73,transparent_45%),linear-gradient(135deg,#07111f,#10223d_45%,#091726)] px-6 py-10 text-slate-100">
      <div className="mx-auto grid max-w-6xl gap-8 lg:grid-cols-[1.2fr_0.8fr]">
        <section className="rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-2xl shadow-sky-950/30 backdrop-blur">
          <p className="text-sm uppercase tracking-[0.4em] text-sky-200/80">PassiveCaptcha</p>
          <h1 className="mt-4 max-w-3xl font-serif text-5xl leading-tight text-white">
            Invisible bot scoring, built around behavioral evidence instead of CAPTCHA friction.
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-slate-300">
            This demo page collects passive client-side signals, extracts behavioral features, and
            produces a live session score. The first real milestone is clean labeled data.
          </p>

          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {featureLabels.map((label) => (
              <div key={label} className="rounded-2xl border border-white/10 bg-slate-950/30 p-4">
                <p className="text-sm text-slate-300">{label}</p>
              </div>
            ))}
          </div>

          <form className="mt-10 grid gap-4 rounded-[1.5rem] border border-white/10 bg-slate-950/40 p-6">
            <div>
              <label className="mb-2 block text-sm text-slate-300" htmlFor="email">
                Work email
              </label>
              <input
                id="email"
                type="email"
                placeholder="analyst@company.com"
                className="w-full rounded-xl border border-white/10 bg-white/10 px-4 py-3 text-white outline-none ring-0 placeholder:text-slate-400"
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
                className="w-full rounded-xl border border-white/10 bg-white/10 px-4 py-3 text-white outline-none ring-0 placeholder:text-slate-400"
              />
            </div>
            <button
              type="button"
              className="rounded-xl bg-sky-300 px-4 py-3 font-semibold text-slate-950 transition hover:bg-sky-200"
            >
              Continue
            </button>
          </form>
        </section>

        <aside className="space-y-6">
          <section className="rounded-[2rem] border border-white/10 bg-slate-950/40 p-6 backdrop-blur">
            <p className="text-sm uppercase tracking-[0.35em] text-slate-400">Live score</p>
            <div className="mt-5 flex items-end justify-between">
              <div>
                <p className="text-6xl font-semibold">
                  {assessment ? assessment.score.toFixed(2) : "0.50"}
                </p>
                <p className="mt-3 text-sm text-slate-400">
                  Uses the FastAPI scorer when available and falls back to the local heuristic if
                  the API is unreachable.
                </p>
              </div>
              <span className={`rounded-full px-4 py-2 text-sm font-medium ${verdictTone}`}>
                {assessment?.verdict ?? "review"}
              </span>
            </div>
          </section>

          <section className="rounded-[2rem] border border-white/10 bg-slate-950/40 p-6 backdrop-blur">
            <p className="text-sm uppercase tracking-[0.35em] text-slate-400">Observed features</p>
            <dl className="mt-5 space-y-3 text-sm text-slate-200">
              <div className="flex items-center justify-between">
                <dt>Samples</dt>
                <dd>{assessment?.features.sampleCount ?? 0}</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt>Mouse linearity</dt>
                <dd>{assessment?.features.mouseLinearity.toFixed(2) ?? "0.00"}</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt>Average key interval</dt>
                <dd>{assessment?.features.averageKeyInterval.toFixed(0) ?? "0"} ms</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt>Time to first interaction</dt>
                <dd>{assessment?.features.timeToFirstInteractionMs ?? 0} ms</dd>
              </div>
            </dl>
          </section>

          <section className="rounded-[2rem] border border-white/10 bg-slate-950/40 p-6 backdrop-blur">
            <p className="text-sm uppercase tracking-[0.35em] text-slate-400">Flagged signals</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {(assessment?.flaggedSignals.length ? assessment.flaggedSignals : ["none"]).map(
                (signal) => (
                  <span
                    key={signal}
                    className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200"
                  >
                    {signal}
                  </span>
                )
              )}
            </div>
          </section>
        </aside>
      </div>
    </main>
  );
}
