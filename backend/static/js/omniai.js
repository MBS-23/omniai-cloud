/* ==========================================================================
   OmniAI Cloud — client
   --------------------------------------------------------------------------
   No framework, no build step. The page is server-rendered; this file handles
   input, the live pipeline stream and result rendering.

   The previous version advanced a progress bar on a setTimeout while a single
   blocking request ran. Progress now comes from the server over Server-Sent
   Events, so what the interface shows is what the pipeline is actually doing.
   ========================================================================== */
(() => {
  'use strict';

  const config = JSON.parse(document.getElementById('omniai-config').textContent);

  const $ = (id) => document.getElementById(id);

  const el = {
    form: $('analysisForm'),
    tabFile: $('tabFile'), tabText: $('tabText'),
    panelFile: $('panelFile'), panelText: $('panelText'),
    dropzone: $('dropzone'), fileInput: $('fileInput'),
    dropzoneTitle: $('dropzoneTitle'), fileMeta: $('fileMeta'),
    filePreview: $('filePreview'),
    textInput: $('textInput'), textMetrics: $('textMetrics'),
    sampleBtn: $('sampleBtn'), clearTextBtn: $('clearTextBtn'),
    autoDetect: $('autoDetect'),
    sourceLanguage: $('sourceLanguage'), targetLanguage: $('targetLanguage'),
    analyzeBtn: $('analyzeBtn'), analyzeLabel: $('analyzeLabel'), resetBtn: $('resetBtn'),
    inputNote: $('inputNote'),
    pipelineSection: $('pipelineSection'), stages: $('stages'),
    progressPct: $('progressPct'), progressStage: $('progressStage'),
    meter: $('meter'), meterFill: $('meterFill'),
    flow: $('flow'), flowLive: $('flowLive'),
    analysing: $('analysing'), analysingStage: $('analysingStage'),
    resultsSkeleton: $('resultsSkeleton'),
    resultsBody: $('resultsBody'), emptyState: $('emptyState'),
    resultActions: $('resultActions'), statusArea: $('statusArea'),
    figures: $('figures'), modelTags: $('modelTags'), panels: $('panels'),
    explainSection: $('explainSection'), explainBody: $('explainBody'),
    expandAllBtn: $('expandAllBtn'), downloadBtn: $('downloadBtn'),
    copyExplainBtn: $('copyExplainBtn'),
    toasts: $('toasts'),
  };

  // Motion helpers from ui.js. The fallbacks keep this file working if that
  // script is blocked or fails: values appear instantly instead of animating.
  const motion = window.OmniAIMotion || {
    observeReveals() {},
    attachTilt() {},
    countUp(el, value, o = {}) {
      el.textContent = Number(value).toFixed(o.decimals || 0) + (o.suffix || '');
    },
    animateBar(fill, percent) { fill.style.width = `${percent}%`; },
  };

  const DRAFT_KEY = 'omniai.draft.v2';
  const SAMPLE_TEXT =
    'కృత్రిమ మేధస్సు ఈరోజు అనేక రంగాలలో ఉపయోగించబడుతోంది. ' +
    'ఇది వైద్యం, విద్య మరియు వ్యవసాయంలో గొప్ప మార్పులు తీసుకువస్తోంది.';

  let mode = 'file';
  let latestResult = null;
  let running = false;

  /* ---------------------------------------------------------- helpers */

  // Every value that reaches the DOM goes through here. Model output is
  // untrusted text and must never be interpolated as markup.
  const text = (value) => document.createTextNode(String(value ?? ''));

  function node(tag, className, textContent) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (textContent !== undefined && textContent !== null) {
      element.textContent = String(textContent);
    }
    return element;
  }

  function icon(name, size = 16) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#${name}`);
    svg.appendChild(use);
    return svg;
  }

  const titleCase = (value) =>
    String(value || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  const languageName = (code) =>
    config.languageNames[String(code || '').toLowerCase()] || code || 'Unknown';

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function toast(message, variant = '') {
    const box = node('div', `toast ${variant ? `toast--${variant}` : ''}`);
    box.setAttribute('role', variant === 'danger' ? 'alert' : 'status');
    box.appendChild(icon(variant === 'danger' ? 'i-alert' : 'i-check'));
    box.appendChild(node('div', '', message));

    const close = node('button', 'toast__close');
    close.type = 'button';
    close.setAttribute('aria-label', 'Dismiss');
    close.appendChild(icon('i-close', 14));
    close.addEventListener('click', () => box.remove());
    box.appendChild(close);

    el.toasts.appendChild(box);
    setTimeout(() => box.remove(), 5000);
  }

  function banner(kind, title, body) {
    const wrap = node('div', `banner banner--${kind}`);
    wrap.setAttribute('role', kind === 'danger' ? 'alert' : 'status');
    const glyph = icon(kind === 'danger' || kind === 'warn' ? 'i-alert' : 'i-info', 18);
    glyph.classList.add('banner__icon');
    wrap.appendChild(glyph);

    const content = node('div');
    if (title) content.appendChild(node('div', 'banner__title', title));
    content.appendChild(text(body));
    wrap.appendChild(content);
    return wrap;
  }

  const clearNode = (target) => { while (target.firstChild) target.removeChild(target.firstChild); };

  /* ------------------------------------------------------- language UI */

  function fillLanguages() {
    const codes = config.translationCodes || [];

    const auto = node('option', '', 'Detect automatically');
    auto.value = 'auto';
    el.sourceLanguage.appendChild(auto);

    codes.forEach((code) => {
      const source = node('option', '', languageName(code));
      source.value = code;
      el.sourceLanguage.appendChild(source);

      const target = node('option', '', languageName(code));
      target.value = code;
      el.targetLanguage.appendChild(target);
    });

    el.sourceLanguage.value = 'auto';
    el.targetLanguage.value = 'en';
  }

  /* --------------------------------------------------------- input UI */

  function setMode(next) {
    mode = next;
    const isFile = next === 'file';

    el.tabFile.setAttribute('aria-selected', String(isFile));
    el.tabText.setAttribute('aria-selected', String(!isFile));
    el.panelFile.hidden = !isFile;
    el.panelText.hidden = isFile;
    el.inputNote.textContent = isFile
      ? 'Choose a source to analyse'
      : `Up to ${config.maxTextChars.toLocaleString()} characters`;
  }

  function updateFileMeta() {
    const file = el.fileInput.files[0];
    clearNode(el.filePreview);
    el.filePreview.hidden = true;

    if (!file) {
      el.dropzone.classList.remove('has-file');
      el.dropzoneTitle.textContent = 'Drop a file here, or click to browse';
      el.fileMeta.textContent = 'No file selected';
      return;
    }

    el.dropzone.classList.add('has-file');
    el.dropzoneTitle.textContent = file.name;
    el.fileMeta.textContent = `${formatBytes(file.size)} · ${file.type || 'unknown type'}`;

    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      const image = node('img');
      image.src = url;
      image.alt = `Preview of ${file.name}`;
      image.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
      el.filePreview.appendChild(image);
      el.filePreview.hidden = false;
    }
  }

  function updateTextMetrics() {
    const value = el.textInput.value;
    const words = value.trim() ? value.trim().split(/\s+/).length : 0;
    el.textMetrics.textContent =
      `${value.length.toLocaleString()} characters · ${words.toLocaleString()} words`;

    if (value.length > config.maxTextChars) {
      el.textMetrics.style.color = 'var(--danger)';
    } else {
      el.textMetrics.style.color = '';
    }

    try { localStorage.setItem(DRAFT_KEY, value); } catch { /* private mode */ }
  }

  /* ------------------------------------------------------- pipeline UI */

  const stageRows = new Map();

  function resetPipeline() {
    stageRows.clear();
    resetFlow();
    clearNode(el.stages);
    el.meterFill.style.width = '0%';
    el.meter.setAttribute('aria-valuenow', '0');
    el.progressPct.textContent = '0%';
    el.progressStage.textContent = 'Starting…';
    el.pipelineSection.hidden = false;
  }

  const STATUS_ICON = {
    start: 'i-dot', done: 'i-check', skipped: 'i-dash', error: 'i-alert',
  };

  // Pipeline stage -> node in the flow diagram. Several model stages share the
  // 'models' node, which is the point: the diagram shows the shape of the
  // pipeline, the list below shows each step.
  const FLOW_NODE = {
    inspect: 'inspect',
    route: 'route',
    image_classification: 'models',
    extract_text: 'models',
    object_detection: 'models',
    wildlife_detection: 'models',
    caption_image: 'models',
    document_ai: 'models',
    translate: 'models',
    analyze_text: 'models',
    explain: 'explain',
  };

  const FLOW_TRACK_LENGTH = 860;

  function resetFlow() {
    if (!el.flow) return;
    el.flow.querySelectorAll('.flow__node').forEach((n) => { delete n.dataset.state; });
    if (el.flowLive) el.flowLive.style.strokeDashoffset = String(FLOW_TRACK_LENGTH);
  }

  function updateFlow(event) {
    if (!el.flow) return;

    if (el.flowLive) {
      const filled = FLOW_TRACK_LENGTH * (1 - Math.min(100, event.percent || 0) / 100);
      el.flowLive.style.strokeDashoffset = String(filled);
    }

    const nodeName = FLOW_NODE[event.stage];
    if (!nodeName) return;
    const node = el.flow.querySelector(`.flow__node[data-node="${nodeName}"]`);
    if (!node) return;

    if (event.status === 'error') {
      node.dataset.state = 'error';
    } else if (event.status === 'start') {
      // Never downgrade a node that already errored.
      if (node.dataset.state !== 'error') node.dataset.state = 'active';
    } else if (node.dataset.state !== 'error') {
      node.dataset.state = 'done';
    }

    // Once any model stage finishes, the aggregate node is in play.
    if (nodeName === 'models' && event.status !== 'start') {
      const aggregate = el.flow.querySelector('.flow__node[data-node="aggregate"]');
      if (aggregate && !aggregate.dataset.state) aggregate.dataset.state = 'active';
    }
    if (event.stage === 'explain' && event.status === 'done') {
      const aggregate = el.flow.querySelector('.flow__node[data-node="aggregate"]');
      if (aggregate && aggregate.dataset.state !== 'error') aggregate.dataset.state = 'done';
    }
  }

  function applyProgress(event) {
    updateFlow(event);
    const percent = Math.max(0, Math.min(100, event.percent || 0));
    el.meterFill.style.width = `${percent}%`;
    el.meter.setAttribute('aria-valuenow', String(percent));
    el.progressPct.textContent = `${percent}%`;
    el.progressStage.textContent = event.label || '';
    if (el.analysingStage) {
      el.analysingStage.textContent = event.detail
        ? `${event.label} — ${event.detail}`
        : (event.label || '');
    }

    let row = stageRows.get(event.stage);
    if (!row) {
      row = node('li', 'stage');
      const glyph = node('span', 'stage__icon');
      row.appendChild(glyph);
      row.appendChild(node('span', 'stage__name', event.label));
      row.appendChild(node('span', 'stage__detail'));
      el.stages.appendChild(row);
      stageRows.set(event.stage, row);
    }

    row.dataset.status = event.status;
    const glyph = row.firstChild;
    clearNode(glyph);
    glyph.appendChild(icon(STATUS_ICON[event.status] || 'i-dot', 14));
    row.lastChild.textContent = event.detail || '';
  }

  /* -------------------------------------------------------- result UI */

  function figure(label, value, accent = false) {
    const box = node('div', `figure reveal${accent ? ' figure--accent' : ''}`);
    box.dataset.tilt = '';

    const cell = node('div', 'figure__value');
    // Numeric figures count up; text figures are written directly.
    if (typeof value === 'number' || /^\d+$/.test(String(value))) {
      cell.textContent = '0';
      motion.countUp(cell, Number(value));
    } else {
      cell.textContent = String(value);
    }

    box.appendChild(cell);
    box.appendChild(node('div', 'figure__label', label));
    motion.attachTilt(box);
    return box;
  }

  function row(key, value) {
    const wrap = node('div', 'row');
    wrap.appendChild(node('div', 'row__key', key));
    const cell = node('div', 'row__val');
    if (value instanceof Node) cell.appendChild(value);
    else cell.appendChild(text(value));
    wrap.appendChild(cell);
    return wrap;
  }

  function confidenceBar(percent, accent = false) {
    const wrap = node('div', 'bar');
    const track = node('div', 'bar__track');
    const fill = node('div', `bar__fill${accent ? ' bar__fill--accent' : ''}`);
    track.appendChild(fill);
    wrap.appendChild(track);

    const value = node('span', 'bar__num', '0.0%');
    wrap.appendChild(value);

    // The bar fills and the number counts when the row scrolls into view.
    motion.animateBar(fill, percent);
    motion.countUp(value, percent, { decimals: 1, suffix: '%' });
    return wrap;
  }

  function panel(title, badge, buildBody, open = true, glyph = 'i-layers') {
    const section = node('section', 'panel rise');

    const head = node('button', 'panel__head');
    head.type = 'button';
    head.setAttribute('aria-expanded', String(open));

    const mark = node('span', 'panel__icon');
    mark.appendChild(icon(glyph, 16));
    head.appendChild(mark);

    head.appendChild(node('span', 'panel__title', title));
    if (badge) head.appendChild(node('span', 'panel__badge', badge));
    const chevron = icon('i-chevron', 16);
    chevron.classList.add('panel__chevron');
    head.appendChild(chevron);

    const body = node('div', 'panel__body');
    body.hidden = !open;
    buildBody(body);

    head.addEventListener('click', () => {
      const expanded = head.getAttribute('aria-expanded') === 'true';
      head.setAttribute('aria-expanded', String(!expanded));
      body.hidden = expanded;
    });

    section.appendChild(head);
    section.appendChild(body);
    return section;
  }

  function renderFigures(data) {
    clearNode(el.figures);
    const meta = data.metadata || {};
    const results = data.results || {};

    el.figures.appendChild(figure('Input type', titleCase(meta.input_type)));
    el.figures.appendChild(
      figure('Models run', String((data.selected_modules || []).length), true)
    );

    if (meta.language) {
      el.figures.appendChild(figure('Language', languageName(meta.language)));
    }
    if (results.image_classification) {
      el.figures.appendChild(
        figure('Classified as', titleCase(results.image_classification.label))
      );
    }
    if (results.object_detection) {
      el.figures.appendChild(
        figure('Objects found', String(results.object_detection.object_count ?? 0))
      );
    }
    if (results.text_analytics) {
      el.figures.appendChild(
        figure('Sentiment', titleCase(results.text_analytics.sentiment))
      );
    }
    if (results.document) {
      el.figures.appendChild(figure('Document', results.document.type));
    }
  }

  function renderModels(data) {
    clearNode(el.modelTags);
    (data.selected_modules || []).forEach((name) => {
      const tag = node('span', 'tag tag--active', titleCase(name));
      el.modelTags.appendChild(tag);
    });
    if (!(data.selected_modules || []).length) {
      el.modelTags.appendChild(node('span', 'tag tag--muted', 'No models executed'));
    }
  }

  function renderPanels(data) {
    clearNode(el.panels);
    const results = data.results || {};

    if (results.image_classification) {
      const c = results.image_classification;
      const ens = c.ensemble || {};
      const badge = ens.agreement === 'agreed' ? '2 models agreed'
        : ens.agreement === 'disagreed' ? 'resolved by reasoning'
        : 'single model';

      el.panels.appendChild(panel('Image classification', badge, (body) => {
        const rows = node('div', 'rows');
        rows.appendChild(row('Label', titleCase(c.label)));
        rows.appendChild(row('Confidence', confidenceBar(c.confidence, true)));
        if (ens.models) rows.appendChild(row('Models', ens.models.join(', ')));
        const runnersUp = (c.top_k || []).slice(1, 4);
        if (runnersUp.length) {
          rows.appendChild(row('Also considered', runnersUp
            .map((k) => `${titleCase(k.label)} ${Number(k.confidence).toFixed(1)}%`)
            .join(' · ')));
        }
        body.appendChild(rows);

        // Show the ensemble's working: what each model said, and how a
        // disagreement was settled. This is the accuracy mechanism, so it
        // should be inspectable rather than hidden.
        if (ens.agreement === 'agreed' && ens.labels) {
          body.appendChild(banner(
            'ok', 'Corroborated',
            `Both models identified this independently (${ens.labels.join(' / ')}), ` +
            'so the confidence above is raised.'
          ));
        } else if (ens.agreement === 'disagreed') {
          const wrap = node('div');
          wrap.appendChild(banner(
            'info', 'Models disagreed',
            `${(ens.labels || []).join(' vs ')} — settled by ${ens.resolved_by || 'the primary model'}.`
          ));
          if (ens.reasoning) {
            const why = node('p', 'note mt-3', ens.reasoning);
            wrap.appendChild(why);
          }
          body.appendChild(node('div', 'stage-spacer'));
          body.appendChild(wrap);
        }
      }, true, 'i-scan'));
    }

    if (results.object_detection) {
      const d = results.object_detection;
      const refined = (d.detections || []).filter((x) => x.corrected).length;
      const badge = `${d.object_count ?? 0} found${refined ? ` · ${refined} refined` : ''}`;

      el.panels.appendChild(panel('Object detection', badge, (body) => {
        if (d.bbox_available === false) {
          body.appendChild(banner(
            'info', null,
            'This backend identifies objects but does not return bounding boxes, ' +
            'so no overlay is shown. The paper profile returns boxes from YOLOv8.'
          ));
          body.appendChild(node('div', 'stage-spacer'));
        }

        if (d.model) {
          const rows = node('div', 'rows mb-5');
          rows.appendChild(row('Model', d.model));
          body.appendChild(rows);
        }

        if (!(d.detections || []).length) {
          body.appendChild(node('p', 'muted', 'No objects were detected in this image.'));
          return;
        }

        const list = node('ul', 'detections');
        d.detections.forEach((item) => {
          const entry = node('li', 'detection');
          const left = node('div');
          left.appendChild(node('div', 'detection__label', titleCase(item.label)));
          if (item.corrected && item.original_yolo_label) {
            left.appendChild(node(
              'span', 'detection__was',
              `refined from "${item.original_yolo_label}"`
            ));
          }
          entry.appendChild(left);
          entry.appendChild(confidenceBar(item.confidence));
          list.appendChild(entry);
        });
        body.appendChild(list);
      }, true, 'i-eye'));
    }

    if (results.caption) {
      const captionModel = results.caption.model || '';
      el.panels.appendChild(panel('Vision–language caption', captionModel.split(' (')[0] || 'BLIP', (body) => {
        body.appendChild(node('blockquote', 'quote', results.caption.caption));
        if (results.caption.original_caption || captionModel) {
          const rows = node('div', 'rows mt-5');
          if (results.caption.original_caption) {
            rows.appendChild(row('Before enrichment', results.caption.original_caption));
          }
          if (captionModel) rows.appendChild(row('Model', captionModel));
          body.appendChild(rows);
        }
      }, true, 'i-spark'));
    }

    if (results.ocr) {
      const o = results.ocr;
      el.panels.appendChild(panel('Extracted text', `${o.word_count || 0} words`, (body) => {
        const rows = node('div', 'rows');
        if (o.language_hint) rows.appendChild(row('Script', o.language_hint));
        rows.appendChild(row('Characters', String((o.text || '').length)));
        if (o.mean_confidence != null) rows.appendChild(row('Confidence', confidenceBar(o.mean_confidence, true)));
        if (o.model) rows.appendChild(row('Model', o.model));
        body.appendChild(rows);
        body.appendChild(node('pre', 'excerpt mt-5', o.text || ''));
      }, true, 'i-text'));
    }

    if (results.document) {
      const doc = results.document;
      el.panels.appendChild(panel('Document analysis', doc.type, (body) => {
        const rows = node('div', 'rows');
        rows.appendChild(row('Extraction', doc.extraction_method));
        if (doc.page_count) rows.appendChild(row('Pages', String(doc.page_count)));
        if (doc.summary_model && doc.summary_generated) rows.appendChild(row('Summariser', doc.summary_model));
        if (doc.coverage_note) rows.appendChild(row('Coverage', doc.coverage_note));
        body.appendChild(rows);

        if (doc.summary) {
          body.appendChild(node('div', 'label mt-6 mb-3', 'Summary'));
          body.appendChild(node('p', 'prose', doc.summary));
        }

        if ((doc.insights || []).length) {
          body.appendChild(node('div', 'label mt-6 mb-3', 'Observations'));
          const list = node('ul', 'steps');
          doc.insights.forEach((line) => list.appendChild(node('li', '', line)));
          body.appendChild(list);
        }

        if (doc.extracted_text) {
          body.appendChild(node('div', 'label mt-6 mb-3', 'Extracted text'));
          body.appendChild(node('pre', 'excerpt', doc.extracted_text));
        }
      }, true, 'i-doc'));
    }

    if (results.translation) {
      const t = results.translation;
      const badge = `${languageName(t.source_language)} → ${languageName(t.target_language)}`;
      el.panels.appendChild(panel('Translation', badge, (body) => {
        body.appendChild(node('blockquote', 'quote', t.translated_text));
        if (t.model) {
          const rows = node('div', 'rows mt-5');
          rows.appendChild(row('Model', t.model));
          body.appendChild(rows);
        }
      }, true, 'i-globe'));
    }

    if (results.text_analytics) {
      const a = results.text_analytics;
      el.panels.appendChild(panel('Text analytics', titleCase(a.sentiment), (body) => {
        const rows = node('div', 'rows');
        rows.appendChild(row('Detected language', languageName(a.original_language)));
        rows.appendChild(row('Sentiment', titleCase(a.sentiment)));
        rows.appendChild(row('Confidence', confidenceBar(a.sentiment_confidence, true)));
        if (a.model) rows.appendChild(row('Models', a.model));
        body.appendChild(rows);

        body.appendChild(node('div', 'label mt-6 mb-3',
          `Named entities (${(a.entities || []).length})`));

        if (!(a.entities || []).length) {
          body.appendChild(node('p', 'muted', 'No named entities were found.'));
        } else {
          const list = node('div', 'entities');
          a.entities.forEach((entity) => {
            const chip = node('span', 'entity');
            chip.appendChild(node('span', 'entity__type', entity.label));
            chip.appendChild(text(entity.text));
            list.appendChild(chip);
          });
          body.appendChild(list);
        }
      }, true, 'i-spark'));
    }

    if (!el.panels.children.length) {
      el.panels.appendChild(banner(
        'warn', 'No model output',
        'The pipeline ran but produced no results. Check the pipeline steps above.'
      ));
    }
  }

  function renderResult(data) {
    latestResult = data;

    clearNode(el.statusArea);
    (data.warnings || []).forEach((warning) => {
      el.statusArea.appendChild(banner('warn', 'Partial result', warning));
    });

    el.emptyState.hidden = true;
    if (el.resultsSkeleton) el.resultsSkeleton.hidden = true;
    el.resultsBody.hidden = false;
    el.resultActions.hidden = false;

    renderFigures(data);
    renderModels(data);
    renderPanels(data);
    motion.observeReveals(el.resultsBody);

    if (data.explanation) {
      clearNode(el.explainBody);
      data.explanation.split('\n').forEach((line) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        if (trimmed.startsWith('-') || trimmed.startsWith('•')) {
          el.explainBody.appendChild(
            node('p', 'mb-3', trimmed.replace(/^[-•]\s*/, '• '))
          );
        } else {
          el.explainBody.appendChild(node('p', '', trimmed));
        }
      });
      el.explainSection.hidden = false;
    }

    el.resultsBody.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderFailure(detail) {
    clearNode(el.statusArea);
    el.statusArea.appendChild(banner('danger', 'Analysis failed', detail));
    if (el.resultsSkeleton) el.resultsSkeleton.hidden = true;
    el.emptyState.hidden = false;
    el.resultsBody.hidden = true;
    el.resultActions.hidden = true;
    toast(detail, 'danger');
  }

  /* ----------------------------------------------------------- submit */

  function setRunning(state) {
    running = state;
    el.analyzeBtn.disabled = state;
    el.analyzeBtn.querySelector('.btn__spinner').hidden = !state;
    el.analyzeLabel.textContent = state ? 'Analysing…' : 'Analyse';

    // The AI core, the scan beam and the skeletons all mean the same thing:
    // work is in flight. They appear and disappear together so motion on the
    // page always corresponds to an open stream.
    if (el.analysing) el.analysing.hidden = !state;
    if (el.filePreview) el.filePreview.classList.toggle('is-scanning', state && mode === 'file');

    if (el.resultsSkeleton) {
      el.resultsSkeleton.hidden = !state;
      if (state) {
        el.emptyState.hidden = true;
        el.resultsBody.hidden = true;
        el.resultActions.hidden = true;
      }
    }
  }

  /* Parse an SSE byte stream incrementally. fetch is used rather than
     EventSource because the request is a POST carrying a file. */
  async function consumeStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let split;
      while ((split = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);

        let name = null;
        let payload = null;
        for (const line of block.split('\n')) {
          if (line.startsWith('event: ')) name = line.slice(7).trim();
          else if (line.startsWith('data: ')) {
            try { payload = JSON.parse(line.slice(6)); } catch { payload = null; }
          }
        }
        if (!name) continue;

        if (name === 'progress' && payload) applyProgress(payload);
        else if (name === 'result' && payload) renderResult(payload);
        else if (name === 'failed' && payload) {
          renderFailure(payload.detail || 'The analysis could not be completed.');
        }
      }
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (running) return;

    let endpoint;
    let body;

    if (mode === 'file') {
      const file = el.fileInput.files[0];
      if (!file) {
        toast('Choose a file to analyse first.', 'danger');
        el.dropzone.focus();
        return;
      }
      endpoint = '/api/analyze/file/stream';
      body = new FormData();
      body.append('file', file);
    } else {
      const value = el.textInput.value.trim();
      if (!value) {
        toast('Enter some text to analyse.', 'danger');
        el.textInput.focus();
        return;
      }
      if (value.length > config.maxTextChars) {
        toast(`Text is too long. The maximum is ${config.maxTextChars.toLocaleString()} characters.`, 'danger');
        return;
      }
      endpoint = '/api/analyze/text/stream';
      body = new FormData();
      body.append('text', value);
      body.append('target_language', el.targetLanguage.value);
      body.append('auto_detect_source', String(el.autoDetect.checked));
      if (!el.autoDetect.checked) {
        body.append('source_language', el.sourceLanguage.value);
      }
    }

    setRunning(true);
    resetPipeline();
    clearNode(el.statusArea);

    try {
      const response = await fetch(endpoint, { method: 'POST', body });

      if (!response.ok) {
        let detail = `Request failed (${response.status}).`;
        try {
          const error = await response.json();
          if (error.detail) detail = error.detail;
          if (response.status === 429 && error.retry_after) {
            detail += ` Try again in ${error.retry_after} seconds.`;
          }
        } catch { /* non-JSON error body */ }
        renderFailure(detail);
        return;
      }

      await consumeStream(response);
    } catch (error) {
      renderFailure('Could not reach the server. Check that it is still running.');
      console.error(error);
    } finally {
      setRunning(false);
    }
  }

  /* ------------------------------------------------------------ wiring */

  function resetAll() {
    el.form.reset();
    updateFileMeta();
    updateTextMetrics();
    el.sourceLanguage.value = 'auto';
    el.targetLanguage.value = 'en';
    el.sourceLanguage.disabled = true;
    el.pipelineSection.hidden = true;
    el.explainSection.hidden = true;
    el.resultsBody.hidden = true;
    el.resultActions.hidden = true;
    el.emptyState.hidden = false;
    clearNode(el.statusArea);
    latestResult = null;
    try { localStorage.removeItem(DRAFT_KEY); } catch { /* ignore */ }
  }

  function init() {
    fillLanguages();

    el.tabFile.addEventListener('click', () => setMode('file'));
    el.tabText.addEventListener('click', () => setMode('text'));

    el.fileInput.addEventListener('change', updateFileMeta);

    ['dragenter', 'dragover'].forEach((name) => {
      el.dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        el.dropzone.classList.add('is-dragging');
      });
    });
    ['dragleave', 'drop'].forEach((name) => {
      el.dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        el.dropzone.classList.remove('is-dragging');
      });
    });
    el.dropzone.addEventListener('drop', (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (!file) return;
      const transfer = new DataTransfer();
      transfer.items.add(file);
      el.fileInput.files = transfer.files;
      updateFileMeta();
    });

    el.textInput.addEventListener('input', updateTextMetrics);
    el.textInput.addEventListener('keydown', (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
        el.form.requestSubmit();
      }
    });

    el.sampleBtn.addEventListener('click', () => {
      el.textInput.value = SAMPLE_TEXT;
      el.targetLanguage.value = 'en';
      updateTextMetrics();
      el.textInput.focus();
    });

    el.clearTextBtn.addEventListener('click', () => {
      el.textInput.value = '';
      updateTextMetrics();
      el.textInput.focus();
    });

    el.autoDetect.addEventListener('change', () => {
      el.sourceLanguage.disabled = el.autoDetect.checked;
    });

    el.resetBtn.addEventListener('click', resetAll);
    el.form.addEventListener('submit', submit);

    el.expandAllBtn.addEventListener('click', () => {
      const heads = el.panels.querySelectorAll('.panel__head');
      const shouldOpen = Array.from(heads).some(
        (head) => head.getAttribute('aria-expanded') === 'false'
      );
      heads.forEach((head) => {
        head.setAttribute('aria-expanded', String(shouldOpen));
        head.nextElementSibling.hidden = !shouldOpen;
      });
      el.expandAllBtn.textContent = shouldOpen ? 'Collapse all' : 'Expand all';
    });

    el.downloadBtn.addEventListener('click', () => {
      if (!latestResult) return;
      const blob = new Blob([JSON.stringify(latestResult, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const link = node('a');
      link.href = url;
      link.download = `omniai-analysis-${Date.now()}.json`;
      link.click();
      URL.revokeObjectURL(url);
      toast('Analysis downloaded.', 'ok');
    });

    el.copyExplainBtn.addEventListener('click', async () => {
      if (!latestResult?.explanation) return;
      try {
        await navigator.clipboard.writeText(latestResult.explanation);
        toast('Explanation copied.', 'ok');
      } catch {
        toast('Copying is not available in this browser.', 'danger');
      }
    });

    try {
      const draft = localStorage.getItem(DRAFT_KEY);
      if (draft) el.textInput.value = draft;
    } catch { /* private mode */ }

    document.querySelectorAll('[data-count]').forEach((element) => {
      const target = Number(element.dataset.count);
      if (!Number.isFinite(target)) return;
      // Hold at zero until the figure is actually on screen.
      if (!('IntersectionObserver' in window)) { motion.countUp(element, target); return; }
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          motion.countUp(element, target, { duration: 1100 });
          observer.unobserve(entry.target);
        });
      }, { threshold: 0.3 });
      observer.observe(element);
    });

    updateTextMetrics();
    updateFileMeta();
    setMode('file');

    if (config.backendReady === false) {
      el.analyzeBtn.setAttribute('aria-disabled', 'true');
      el.analyzeBtn.title = 'No model backend is configured';
    } else {
      watchWarmup();
    }
  }

  // After a cold start the local models load in the background. Say so in
  // the header chip, and how far along it is, until they are all in memory.
  // A request made meanwhile still works: it waits for the model it needs.
  function watchWarmup() {
    const chip = document.getElementById('backendChip');
    const label = chip && chip.querySelector('.chip__label');
    if (!label) return;

    const tick = async () => {
      let body;
      try {
        const response = await fetch('/api/health', { cache: 'no-store' });
        body = await response.json();
      } catch (error) {
        return; // Offline or restarting: leave the chip as the server rendered it.
      }
      const models = body && body.models;
      if (!models) return; // A hosted backend has nothing to load.

      const base = config.stackLabel || 'Models';
      if (models.loading) {
        chip.classList.add('chip--warming');
        label.textContent = `${base} · loading, ${models.ready} ready`;
        window.setTimeout(tick, 4000);
      } else {
        chip.classList.remove('chip--warming');
        label.textContent = models.ready ? `${base} · ${models.ready} ready` : `${base} · load on first use`;
      }
    };
    tick();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
