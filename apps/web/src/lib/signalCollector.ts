import { PassiveSignals, KeyboardSample, PointerSample, ScrollSample } from "../types";

const MAX_SAMPLES = 200;

function createSessionId(): string {
  return `sess_${Math.random().toString(36).slice(2, 10)}`;
}

function limitPush<T>(array: T[], value: T): void {
  if (array.length >= MAX_SAMPLES) {
    array.shift();
  }
  array.push(value);
}

function getWebGLRenderer(): string | null {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl");

  if (!gl) {
    return null;
  }

  const extension = gl.getExtension("WEBGL_debug_renderer_info");
  if (!extension) {
    return null;
  }

  return gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) as string;
}

function hashCanvasFingerprint(): string {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return "canvas-unavailable";
  }

  ctx.textBaseline = "top";
  ctx.font = "14px monospace";
  ctx.fillStyle = "#123456";
  ctx.fillRect(4, 4, 120, 24);
  ctx.fillStyle = "#f2f2f2";
  ctx.fillText("PassiveCaptcha", 8, 8);

  const data = canvas.toDataURL();
  let hash = 0;
  for (let index = 0; index < data.length; index += 1) {
    hash = (hash << 5) - hash + data.charCodeAt(index);
    hash |= 0;
  }
  return String(hash);
}

export class SignalCollector {
  private signals: PassiveSignals;
  // tracks the timestamp of each keydown so we can compute hold duration on keyup
  private pendingKeyDowns: Map<string, number> = new Map();

  constructor() {
    this.signals = {
      sessionId: createSessionId(),
      startedAt: Date.now(),
      pointer: [],
      keyboard: [],
      scroll: [],
      visibilityChanges: 0,
      visibilityChangeTimes: [],
      firstInteractionAt: null,
      screen: {
        width: window.screen.width,
        height: window.screen.height,
        devicePixelRatio: window.devicePixelRatio
      },
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      platform: navigator.platform || "unknown",
      userAgent: navigator.userAgent || "unknown",
      language: navigator.language || "unknown",
      touchPoints: navigator.maxTouchPoints || 0,
      webglRenderer: getWebGLRenderer() || "",
      canvasHash: hashCanvasFingerprint() || ""
    };
  }

  private markInteraction(timestamp: number): void {
    if (this.signals.firstInteractionAt === null) {
      this.signals.firstInteractionAt = timestamp;
    }
  }

  private recordPointer(sample: PointerSample): void {
    this.markInteraction(sample.t);
    limitPush(this.signals.pointer, sample);
  }

  private recordKeyboard(sample: KeyboardSample): void {
    this.markInteraction(sample.t);
    limitPush(this.signals.keyboard, sample);
  }

  private recordScroll(sample: ScrollSample): void {
    this.markInteraction(sample.t);
    limitPush(this.signals.scroll, sample);
  }

  start(): () => void {
    const onPointerMove = (event: MouseEvent) => {
      this.recordPointer({ x: event.clientX, y: event.clientY, t: Date.now(), type: "move" });
    };

    const onPointerDown = (event: MouseEvent) => {
      this.recordPointer({ x: event.clientX, y: event.clientY, t: Date.now(), type: "down" });
    };

    const onPointerUp = (event: MouseEvent) => {
      this.recordPointer({ x: event.clientX, y: event.clientY, t: Date.now(), type: "up" });
    };

    const onClick = (event: MouseEvent) => {
      this.recordPointer({ x: event.clientX, y: event.clientY, t: Date.now(), type: "click" });
    };

    const onEnter = (event: MouseEvent) => {
      this.recordPointer({ x: event.clientX, y: event.clientY, t: Date.now(), type: "enter" });
    };

    const onKeyDown = (event: KeyboardEvent) => {
      const t = Date.now();
      // record the down timestamp so keyup can compute hold duration
      this.pendingKeyDowns.set(event.key, t);
      this.recordKeyboard({ key: event.key, t, type: "down" });
    };

    const onKeyUp = (event: KeyboardEvent) => {
      const t = Date.now();
      const downTime = this.pendingKeyDowns.get(event.key);
      // hold = ms the key was physically held; undefined if we missed the keydown
      const hold = downTime !== undefined ? t - downTime : undefined;
      this.pendingKeyDowns.delete(event.key);
      this.recordKeyboard({ key: event.key, t, type: "up", hold });
    };

    const onScroll = () => {
      this.recordScroll({ x: window.scrollX, y: window.scrollY, t: Date.now() });
    };

    const onVisibility = () => {
      const t = Date.now();
      this.signals.visibilityChanges += 1;
      // keep a full timestamp log so features.ts can compute inter-change timing
      this.signals.visibilityChangeTimes.push(t);
    };

    window.addEventListener("mousemove", onPointerMove, { passive: true });
    window.addEventListener("mousedown", onPointerDown, { passive: true });
    window.addEventListener("mouseup", onPointerUp, { passive: true });
    window.addEventListener("click", onClick, { passive: true });
    window.addEventListener("mouseenter", onEnter, { passive: true });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      window.removeEventListener("mousemove", onPointerMove);
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("mouseup", onPointerUp);
      window.removeEventListener("click", onClick);
      window.removeEventListener("mouseenter", onEnter);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }

  snapshot(): PassiveSignals {
    return {
      ...this.signals,
      pointer: [...this.signals.pointer],
      keyboard: [...this.signals.keyboard],
      scroll: [...this.signals.scroll],
      visibilityChangeTimes: [...this.signals.visibilityChangeTimes],
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      }
    };
  }
}
