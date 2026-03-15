import { FeatureVector, SessionAssessment } from "./types";
import { useEffect, useState } from "react";

export function AnalyticsDashboard({
  assessment,
  features,
}: {
  assessment: SessionAssessment | null;
  features: FeatureVector | null;
}) {
  const [history, setHistory] = useState<number[]>([]);

  useEffect(() => {
    if (assessment) {
      setHistory((prev) => [...prev.slice(-30), assessment.score]);
    }
  }, [assessment]);

  return (
    <section className="col-span-full mt-8 rounded-[2rem] border border-white/10 bg-slate-950/40 p-6 backdrop-blur">
      <h2 className="mb-4 text-sm uppercase tracking-[0.35em] text-slate-400">Live Analytics Dashboard</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Chart Concept */}
        <div className="col-span-1 rounded-xl border border-white/5 bg-black/20 p-4">
          <h3 className="mb-3 text-xs font-semibold text-slate-300">Score History (Live)</h3>
          <div className="flex h-32 items-end justify-between gap-1">
            {history.map((h, i) => (
              <div 
                key={i} 
                className="w-full bg-sky-400/50 rounded-t-sm transition-all" 
                style={{ height: `${h * 100}%` }}
              ></div>
            ))}
          </div>
        </div>

        {/* Feature Breakdown */}
        <div className="col-span-1 rounded-xl border border-white/5 bg-black/20 p-4">
          <h3 className="mb-3 text-xs font-semibold text-slate-300">Signal Diagnostics</h3>
          <ul className="text-xs text-slate-400 space-y-2">
            <li>Mouse Linearity: {features?.mouseLinearity?.toFixed(2) ?? 'N/A'}</li>
            <li>Click Count: {features?.clickCount?.toFixed(0) ?? 'N/A'}</li>
            <li>Key Events: {features?.keyEvents?.toFixed(0) ?? 'N/A'}</li>
            <li>Scroll Events: {features?.scrollEvents?.toFixed(0) ?? 'N/A'}</li>
            <li>Flags: {assessment?.flaggedSignals?.length ?? 0} active</li>
          </ul>
        </div>
        
        {/* Timeline */}
        <div className="col-span-1 rounded-xl border border-white/5 bg-black/20 p-4">
          <h3 className="mb-3 text-xs font-semibold text-slate-300">Flagged Timeline</h3>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {assessment?.flaggedSignals?.length ? (
              assessment.flaggedSignals.map((flag, idx) => (
                <div key={idx} className="text-xs flex gap-2">
                  <span className="text-rose-400 font-bold">●</span>
                  <span className="text-slate-300">{flag}</span>
                </div>
              ))
            ) : (
              <div className="text-xs text-emerald-400">All sequences normal</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
