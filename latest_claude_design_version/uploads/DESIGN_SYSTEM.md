# docs/DESIGN_SYSTEM.md

Source of truth for visual and interaction design in Cinemateca-imgsearch v0.3.0.
This file is read by Claude Code before any HTML/CSS work. Update it when a new
pattern is established in code — do not let the implementation drift from it.

---

## Design brief

**Character:** Cinemateca modernist — high contrast, restrained, institutional.
Reference points: Mubi, Criterion Channel, BFI catalogue.
Not: a SaaS dashboard, not a media player, not a consumer app.

**Primary surface:** Dark. Curators work in darkened rooms. Light mode is not planned for v0.3.0.

**Principle:** The interface is a tool for professionals. It should disappear in service
of the material — film titles, keyframes, scene descriptions. No decorative chrome.

---

## Color palette

All values are CSS custom properties defined in `web/static/css/main.css`.
Never hardcode hex in templates — always use these variables.

### Base surfaces

```css
--c-bg-base:       #1C1A18;   /* page background — dark warm, +15% from original */
--c-bg-raised:     #272421;   /* cards, sidebar — slightly lifted */
--c-bg-overlay:    #312E2B;   /* hover states, selected items */
--c-bg-border:     #3D3A36;   /* dividers, borders */
```

### Text

```css
--c-text-primary:  #F0EDE8;   /* main text — warm off-white, contrast ~17:1 */
--c-text-secondary:#A09890;   /* labels, metadata */
--c-text-muted:    #6E6760;   /* placeholders, disabled */
```

### Accent — Celluloid Amber

The single accent color. Used for: active tab indicator, selected tree item,
focus rings, progress fill, primary action buttons.

```css
--c-accent:        #C8963E;   /* amber — aged celluloid, main accent */
--c-accent-dim:    #8B6520;   /* amber, dimmed — hover states */
--c-accent-text:   #F5C97A;   /* amber, light — text on dark bg with amber context */
```

### Semantic

```css
--c-success:       #4E9E6B;   /* pipeline step: done */
--c-success-bg:    #0D2318;
--c-error:         #C25450;
--c-error-bg:      #2A100F;
--c-processing:    #5B8FCC;   /* pipeline step: in progress */
--c-processing-bg: #0A1929;
--c-pending:       #645E58;   /* pipeline step: not started */
```

---

## Typography

Three fonts, each with a fixed assignment. Never use a font outside its role.

| Variable | Family | Fallback | License | Self-hosted |
|---|---|---|---|---|
| `--font-serif` | IBM Plex Serif | Georgia, serif | Apache 2.0 | `web/static/fonts/` |
| `--font-sans` | IBM Plex Sans | system-ui, sans-serif | Apache 2.0 | `web/static/fonts/` |
| `--font-mono` | iA Writer Mono | Courier New, monospace | OFL | `web/static/fonts/` |

```css
--font-serif: 'IBM Plex Serif', Georgia, serif;
--font-sans:  'IBM Plex Sans', system-ui, sans-serif;
--font-mono:  'iA Writer Mono', 'Courier New', monospace;
```

### Font role assignments

**IBM Plex Serif** — film titles only.
- Sidebar tree: film name (folder names stay sans)
- Scene header when a scene is open
- Contextual subtitle line ("Jeca Tatu (1959) · 247 cenas catalogadas")
- Never used for navigation, buttons, or body text

**IBM Plex Sans** — all UI chrome.
- Tab labels, section headings, panel headers
- Scene descriptions, metadata labels, tags
- Buttons, inputs (container text), placeholders
- Folder names in sidebar tree
- Version numbers, badges, counts

**iA Writer Mono** — query input and timecodes only.
- The text the user types in the search bar (`<input>` value)
- Timecodes in scene card metadata strip (`01:23`, `14:08`)
- Rationale: signals "machine-readable identifier / query string";
  creates semantic distinction between authored content and UI labels
- At 13–13.5px with `letter-spacing: 0.01em` on dark background

### Typographic scale

| Role | Font | Size | Weight | Line-height |
|---|---|---|---|---|
| Film title (sidebar) | Serif | 0.8125rem | 400 | 1.3 |
| Film title (header) | Serif | 1.25rem | 400 | 1.3 |
| Section heading | Sans | 0.9375rem | 500 | 1.4 |
| Body / description | Sans | 0.875rem | 400 | 1.6 |
| Label / caption | Sans | 0.75rem | 400 | 1.4 |
| Micro | Sans | 0.6875rem | 400 | 1.3 |
| Query input | Mono | 0.84375rem | 400 | — |
| Timecode | Mono | 0.6875rem | 400 | — |

---

## Spacing

Base unit: 4px. All spacing is a multiple of this unit.

```
4px   — icon-to-label gap, tight inline spacing
8px   — intra-component padding, gap between sibling elements
12px  — component internal padding (small)
16px  — component internal padding (standard), section gap
24px  — between distinct sections
32px  — between major layout regions
```

---

## Layout

### Shell

Two-column: sidebar (220px fixed) + main (fluid).
No responsive breakpoints in v0.3.0 — minimum viewport 1024px.

```
+--sidebar (220px)--+--main (fluid)--+
| logo + locale     | tab bar         |
| filter input      | tab content     |
| file tree         |                 |
| ...               |                 |
| about link        |                 |
+-------------------+-----------------+
```

### Sidebar

- Background: `--c-bg-raised`
- Right border: `1px solid var(--c-bg-border)`
- Padding: `16px 12px`
- The sidebar is the **primary navigation surface** — not the tab bar.
  The tab bar contextualizes the selected film. The sidebar selects the film.

### Tab bar

- 3 permanent tabs: **Buscar / Search**, **Cenas / Scenes**, **Anotar / Annotate**
- 1 conditional tab: **Processamento / Processing** — renders only when active jobs exist.
  When visible, it carries a count badge (`1`, `2` etc).
- Active tab: bottom border `2px solid var(--c-accent)`, text `var(--c-text-primary)`
- Inactive tab: no border, text `var(--c-text-secondary)`
- Tab height: 40px. Font: sans 0.9375rem/500.
- Processing tab badge: `background: var(--c-processing-bg)`, `color: var(--c-processing)`,
  `font-size: 0.6875rem`, `border-radius: 999px`, `padding: 0 5px`.

---

## Components

### Sidebar tree node

State variants: default, hover, selected (active film).

```
Default:
  padding: 4px 6px
  border-radius: 6px
  color: --c-text-secondary
  background: transparent

Hover:
  background: --c-bg-overlay

Selected (active film):
  background: rgba(200, 150, 62, 0.12)   /* amber at 12% opacity */
  color: --c-accent-text
  border-left: 2px solid var(--c-accent)
  padding-left: 4px                       /* compensate for border */
```

Film title in tree uses `--font-serif`. Folder names use `--font-sans`.

Processing indicator: italic text + spinner icon (vendored Phosphor
`spinner-gap`, regular weight, animated via `animation: spin 1s linear infinite`
through the `spin` class on the `icon()` macro).

### Scene card

Used in Cenas tab and Search results grid. 3-column grid, `gap: 10px`.

```
Card:
  background: --c-bg-raised
  border: 1px solid var(--c-bg-border)
  border-radius: 8px
  overflow: hidden

Keyframe area:
  aspect-ratio: 4/3
  background: --c-bg-base     /* fallback while image loads */
  object-fit: cover

Metadata strip:
  padding: 8px
  font-size: 0.75rem          /* label scale */
  color: --c-text-secondary
  display: flex
  gap: 8px
```

Metadata strip shows: timecode · environment · person count.
Timecode uses `--font-mono` at `0.6875rem`. Environment and count use `--font-sans`.
On hover: card border transitions to `var(--c-bg-border)` brightened → `rgba(200,150,62,0.3)`.

### Search input

Full-width, single bar with secondary mode toggle (text / image).

```
Container:
  background: --c-bg-raised
  border: 1px solid var(--c-bg-border)
  border-radius: 8px
  padding: 10px 14px
  display: flex, align-items: center, gap: 10px

Input text (what the user types):
  font-family: var(--font-mono)
  font-size: 0.84375rem
  letter-spacing: 0.01em
  color: var(--c-text-primary)

Placeholder text:
  font-family: var(--font-sans)
  color: var(--c-text-muted)

On focus:
  border-color: var(--c-accent)
  outline: none

Mode toggle:
  position: right of input
  border-left: 1px solid var(--c-bg-border)
  padding-left: 12px
  font-size: 0.75rem
  color: --c-text-muted
  font-family: var(--font-sans)
  active mode: color var(--c-accent)
```

### Processing tab — pipeline stepper

5 steps: Frames · Cenas · Visual · Embeddings · Descrições.
Displayed as a horizontal row of state pills.

States:
- **done** — `background: var(--c-success-bg)`, `color: var(--c-success)`, checkmark icon
- **active** — `background: var(--c-processing-bg)`, `color: var(--c-processing)`, spinner icon
- **pending** — `background: var(--c-bg-overlay)`, `color: var(--c-pending)`, circle icon

Progress bar:
- Track: `background: var(--c-bg-overlay)`, `height: 5px`, `border-radius: 999px`
- Fill: `background: var(--c-accent)`, animated via width transition

### About modal

Triggered by link in sidebar footer. Rendered as an HTMX-swapped overlay.

```
Backdrop:
  position: fixed (rendered via full-viewport wrapper in flow — see CLAUDE.md note)
  background: rgba(15, 14, 13, 0.85)

Dialog:
  background: --c-bg-raised
  border: 1px solid var(--c-bg-border)
  border-radius: 10px
  padding: 32px
  max-width: 520px
  margin: auto

Header: film-reel icon (vendored Phosphor, regular weight, enlarged via the
`modal-logo` class) + project name in serif
Body: version, license, institutional credits, model attributions
Footer: link to GitHub, link to SETUP.md
```

### Buttons

Primary (used only for "Start processing"):
```
background: var(--c-accent)
color: #1C1A18
border: none
border-radius: 6px
padding: 8px 16px
font: sans 0.875rem/500
```

Secondary (cancel, secondary actions):
```
background: transparent
border: 1px solid var(--c-bg-border)
color: var(--c-text-secondary)
border-radius: 6px
padding: 8px 16px
```

Ghost (inline, low-emphasis):
```
background: transparent
border: none
color: var(--c-text-secondary)
padding: 4px 8px
```

No other button variants in v0.3.0. If you need a new one, document it here first.

---

## Icons

**Library:** Phosphor Icons — path data vendored locally as inline SVG.
**Source:** the official Phosphor **regular** set, vendored verbatim under
`web/static/icons/phosphor/` (with the upstream MIT `LICENSE` for provenance).
**Load:** none. There is **no external CDN** — no
`<script src="https://unpkg.com/@phosphor-icons/web">`, no runtime web
component. This is a deliberate offline-first constraint (a regression test
forbids any external icon reference).

Icons render through the `icon(name, extra)` Jinja macro in
`web/templates/partials/_icons.html`, which emits an inline
`<svg ... viewBox="0 0 256 256" fill="currentColor">`. Two parity properties
matter:

- `fill="currentColor"` — the glyph inherits the surrounding text color, the
  same as the old web component did.
- `class="ph-icon"` sizes the glyph at `1em` (see `main.css`), so it scales
  with the local font-size; the `extra` argument carries through state
  classes (e.g. `spin`, `modal-logo`) for callers that need a larger glyph
  or animation.

Usage (import once per template, then call the macro):

```jinja
{% from "partials/_icons.html" import icon %}
{{ icon("magnifying-glass") }}
{{ icon("film-reel", "modal-logo") }}
{{ icon("spinner-gap", "spin") }}
```

### Canonical icon assignments

Only the **regular** weight is vendored. This is a deliberate offline
tradeoff: a few elements that previously used `fill` or `duotone` weight
(stepper done/error states, the modal logo) are now rendered at `regular`
weight like everything else. There is no per-icon weight choice — color and
emphasis come from the surrounding context (text color, size class), not
from a Phosphor weight variant.

| Element | Phosphor name |
|---|---|
| Library / Acervo | `film-reel` |
| Film / Filme | `film-strip` |
| Search / Buscar | `magnifying-glass` |
| Scenes / Cenas | `squares-four` |
| Annotate / Anotar | `tag` |
| Processing | `spinner-gap` (+ `spin` class) |
| Done / success | `check-circle` |
| Error | `x-circle` |
| Pending | `circle` / `circle-dashed` |
| Not allowed | `prohibit` |
| Remove | `minus-circle` |
| Add film | `upload-simple` |
| About | `info` |
| Locale switch | `globe` |
| Close modal | `x` |

Do not use icons not listed here without updating this table **and**
adding the path data to the `_icons.html` macro (icons are vendored, not
loaded on demand).

---

## i18n conventions

Locale files: `web/locales/pt_BR/LC_MESSAGES/messages.po` and `en/LC_MESSAGES/messages.po`.
Extraction: `pybabel extract -F web/babel.cfg -o web/locales/messages.pot web/`
All user-visible strings wrapped in `{{ _("key") }}` in Jinja templates.

Translation key conventions:
- Keys are the English string (Babel default): `_("Search")`
- For strings with context: `_("%(count)s scenes", count=n)`
- Never construct strings by concatenation — use format strings so translators can reorder.

### Core string pairs

| PT | EN |
|---|---|
| Acervo | Library |
| Buscar | Search |
| Cenas | Scenes |
| Anotar | Annotate |
| Processamento | Processing |
| Sobre | About |
| Adicionar filme | Add film |
| Filtrar acervo | Filter library |
| Iniciar processamento | Start processing |
| Cancelar | Cancel |
| Concluído | Done |
| cenas catalogadas | scenes catalogued |
| restante(s) | remaining |

---

## Motion

Minimal. Dark professional interfaces use motion to communicate state, not to delight.

- **HTMX swaps:** `transition: opacity 150ms ease` via `.htmx-swapping { opacity: 0 }` and `.htmx-settling { opacity: 1 }`.
- **Progress bar fill:** `transition: width 300ms ease-out`
- **Spinner:** `@keyframes spin { to { transform: rotate(360deg) } }` at `1s linear infinite`
- **Modal open:** `transition: opacity 200ms ease`
- **Tree expand/collapse:** no animation in v0.3.0 — instant swap.

No other animations. If you think you need one, document it here first.

---

## What not to do

- No light backgrounds anywhere in v0.3.0.
- No rounded corners above `border-radius: 10px`.
- No gradient backgrounds.
- No box shadows (except focus rings: `box-shadow: 0 0 0 2px var(--c-accent)`).
- No color-coding beyond the semantic palette above.
- No more than three font sizes on any single screen.
- No decorative dividers or ornament.
- No icons outside the canonical assignments table without updating this doc
  AND vendoring the path data into `_icons.html`.
- No external icon CDN / web component (offline-first; a test forbids it).

---

*Last updated: v2 — lighter base surfaces (+15% lightness), three-font system
(iA Writer Mono added for query input and timecodes). Update when any pattern changes in code.*
