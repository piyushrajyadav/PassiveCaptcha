import { useState, useEffect } from 'react';

export function ChallengeOverlay({ onComplete }: { onComplete: (success: boolean) => void }) {
  const [position, setPosition] = useState({ top: 50, left: 50 });

  useEffect(() => {
    const interval = setInterval(() => {
      setPosition({
        top: Math.random() * 80 + 10,
        left: Math.random() * 80 + 10,
      });
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  const handleClick = () => {
    onComplete(true);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="relative h-64 w-96 rounded-xl border border-white/10 bg-slate-950 p-6 shadow-2xl">
        <h2 className="mb-2 text-xl font-bold text-white">Verification Required</h2>
        <p className="mb-4 text-sm text-slate-400">
          We detected suspicious activity. Please click the moving dot to verify you are human.
        </p>
        <div className="relative h-40 w-full overflow-hidden rounded-lg bg-white/5 border border-white/5 cursor-crosshair">
          <button
            onClick={handleClick}
            style={{ top: `${position.top}%`, left: `${position.left}%` }}
            className="absolute h-8 w-8 -translate-x-1/2 -translate-y-1/2 transform rounded-full bg-sky-400 shadow-[0_0_15px_rgba(56,189,248,0.5)] transition-all duration-1000 hover:bg-sky-300"
          />
        </div>
      </div>
    </div>
  );
}
