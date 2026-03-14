/**
 * PassiveCaptcha Enterprise SDK
 * Use this script to integrate the PassiveCaptcha verification layer inside your platform.
 */

const PassiveCaptcha = (function () {
  let sessionToken = null;
  let apiUrl = "http://localhost:8000";
  let fallbackCallback = null;

  async function init(config) {
    if (config?.apiUrl) apiUrl = config.apiUrl;
    if (config?.onChallenge) fallbackCallback = config.onChallenge;

    try {
      const response = await fetch(`${apiUrl}/v1/init`, { method: "POST" });
      const data = await response.json();
      sessionToken = data.token;
      
      console.log("[PassiveCaptcha] Initialized. Session Token:", sessionToken);
      _startCollection();
    } catch (e) {
      console.error("[PassiveCaptcha] SDK Init Error", e);
    }
  }

  function _startCollection() {
    // Stub implementation: Collect signals and send to the ingest endpoint
    setInterval(async () => {
      if (!sessionToken) return;

      const mockPayload = {
        userAgent: navigator.userAgent,
        timestamp: Date.now(),
        screenViewportRatio: window.innerWidth / window.screen.width,
        devicePixelRatio: window.devicePixelRatio,
        hardwareConcurrency: navigator.hardwareConcurrency,
        touchPoints: navigator.maxTouchPoints,
        hasWebGL: 1,
        // ... (other signals collected passively)
      };

      try {
        const response = await fetch(`${apiUrl}/v1/ingest`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${sessionToken}`
          },
          body: JSON.stringify(mockPayload)
        });
        
        const result = await response.json();
        
        // Trigger fallback validation if confidence is low
        if (result.verdict === "review" && fallbackCallback) {
          fallbackCallback();
        }

      } catch (e) {
        // Silently fail to minimize user interruption
      }
    }, 5000);
  }

  return { init };
})();

export default PassiveCaptcha;
