// Inline SVG icons (Lucide-style, 24x24 viewBox, currentColor stroke). Replaces emoji glyphs so
// the clinical UI renders consistently across platforms and stays accessible (each icon is
// aria-hidden; the interactive element carries the label). See ui-ux-pro-max: "no emoji as icons".
const base = {
  width: 20, height: 20, viewBox: "0 0 24 24", fill: "none",
  stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round",
  strokeLinejoin: "round", "aria-hidden": true, focusable: false,
};

export const IconMic = (p) => (
  <svg {...base} {...p}>
    <rect x="9" y="2" width="6" height="12" rx="3" />
    <path d="M5 10a7 7 0 0 0 14 0M12 17v4M8 21h8" />
  </svg>
);

export const IconStop = (p) => (
  <svg {...base} {...p}><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
);

export const IconSend = (p) => (
  <svg {...base} {...p}><path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z" /></svg>
);

export const IconShield = (p) => (
  <svg {...base} {...p}>
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <path d="m9 12 2 2 4-4" />
  </svg>
);

export const IconAlert = (p) => (
  <svg {...base} {...p}>
    <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
    <path d="M12 9v4M12 17h.01" />
  </svg>
);

export const IconPill = (p) => (
  <svg {...base} {...p}>
    <path d="M10.5 20.5 3.5 13.5a5 5 0 0 1 7-7l7 7a5 5 0 0 1-7 7z" />
    <path d="m8.5 8.5 7 7" />
  </svg>
);

export const IconBack = (p) => (
  <svg {...base} {...p}><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
);

export const IconSearch = (p) => (
  <svg {...base} {...p}><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>
);

export const IconChevron = (p) => (
  <svg {...base} {...p}><path d="m9 18 6-6-6-6" /></svg>
);

export const IconActivity = (p) => (
  <svg {...base} {...p}><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
);

export const IconClock = (p) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>
);
