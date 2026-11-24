"""Render a fixed, escaped experiment report from the saved local result."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from webapp import classifier, tabular


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _english_note(value):
    """Render common notes from earlier Chinese-language runs in English."""
    text = str(value)
    translations = {
        '正則化參數只影響線性模型；Random Forest 不使用此參數。': 'Regularization affects linear models only; Random Forest does not use it.',
        '檢查測試錯誤與資料代表性；若依此調整模型，請使用新的獨立測試資料再次評估。': 'Review errors and data representativeness; use new independent test data after further tuning.',
        '此分數來自本機保留的測試集，並非 Kaggle 排行榜分數；沒有提交比賽。': 'Scores use a local holdout, not the Kaggle leaderboard; no submission was made.',
        '可能是唯一識別碼；請確認是否需要排除，以避免評估失真。': 'may be a unique identifier; review whether it should be excluded.',
        '有較多不同文字，將作為文字特徵；請確認不是識別碼或資料洩漏欄位。': 'has many distinct text values; review for identifiers and leakage.',
        '原始方案：': 'Original plan: ', ' 最終方案已排除 ': ' Final plan excluded ',
    }
    for original, english in translations.items():
        text = text.replace(original, english)
    return text


def _chart(title, rows, *, unit='', fixed_range=None, note='', precision=3, paired=False):
    """Render signed values on an explicit common scale; missing data is omitted."""
    rows = [(str(label), value) for label, value in rows if _finite(value)]
    if not rows:
        return ''
    low, high = fixed_range if fixed_range else (min(0, min(value for _, value in rows)), max(0, max(value for _, value in rows)))
    if high <= low:
        high = low + 1
    zero = max(0, min(100, (0 - low) / (high - low) * 100))
    parts = []
    for label, value in rows:
        end = max(0, min(100, (value - low) / (high - low) * 100))
        left, width = min(zero, end), abs(end - zero)
        parts.append(f'<div class="viz-row"><span class="viz-label">{html.escape(label)}</span><div class="viz-track" aria-hidden="true"><i class="viz-zero" style="left:{zero:.2f}%"></i><i class="viz-bar" style="left:{left:.2f}%;width:{width:.2f}%"></i></div><b>{value:.{precision}f}{html.escape(unit)}</b></div>')
    bounds = f'Common scale: {low:.{precision}f} to {high:.{precision}f}{html.escape(unit)}'
    return f'<figure class="viz{" paired" if paired else ""}"><figcaption>{html.escape(title)}</figcaption><p class="muted">{bounds}。{html.escape(note)}</p>{"".join(parts)}</figure>'


def _split_chart(split):
    names = {'train': 'Training set', 'validation': 'Validation set', 'test': 'Test set', 'exploration': 'Exploration data'}
    entries = [(names.get(key, str(key)), value) for key, value in split.items() if _finite(value) and value >= 0]
    total = sum(value for _, value in entries)
    if not total:
        return ''
    segments = ''.join(f'<span class="split-segment split-{index % 4}" style="width:{value / total * 100:.3f}%" title="{html.escape(name)}：{int(value)} rows"></span>' for index, (name, value) in enumerate(entries))
    legend = ''.join(f'<span><i class="split-dot split-{index % 4}"></i>{html.escape(name)} {int(value)} rows（{value / total * 100:.1f}%）</span>' for index, (name, value) in enumerate(entries))
    return f'<figure class="viz"><figcaption>Data split</figcaption><div class="split-bar" aria-hidden="true">{segments}</div><div class="split-legend">{legend}</div></figure>'


def _paired_chart(title, before, after, *, fixed_range=None, unit='', note=''):
    """Compare matched candidates without inventing absent measurements."""
    after_by_model = {item.get('model'): item for item in after if isinstance(item, dict)}
    rows = []
    for item in before:
        if not isinstance(item, dict):
            continue
        other = after_by_model.get(item.get('model'))
        if other and _finite(item.get('value')) and _finite(other.get('value')):
            rows.extend([(str(item.get('model')) + ' · Before', item['value']), (str(item.get('model')) + ' · After', other['value'])])
    return _chart(title, rows, fixed_range=fixed_range, unit=unit, note=note, paired=True)


def _loss_charts(attempts):
    """Show recorded measurements only, grouped by metric and step unit."""
    groups, unavailable = {}, []
    for attempt in attempts:
        name = str(attempt.get('model', '—'))
        curve = attempt.get('loss_curve')
        if not isinstance(curve, dict):
            unavailable.append(name)
            continue
        points = [(point['step'], point['loss']) for point in curve.get('points', [])
                  if isinstance(point, dict) and _finite(point.get('step')) and _finite(point.get('loss')) and point['loss'] >= 0]
        if not points:
            unavailable.append(name)
            continue
        key = (str(curve.get('metric', 'loss')), str(curve.get('step_unit', 'step')), curve.get('sample_count'))
        groups.setdefault(key, []).append((name, sorted(points)))
    figures = []
    palette = ['#7351bc', '#148780', '#bd6c36', '#bd5178', '#5572b5', '#718332']
    for (metric, unit, sample_count), series in groups.items():
        all_points = [point for _, points in series for point in points]
        min_step, max_step = min(p[0] for p in all_points), max(p[0] for p in all_points)
        max_loss = max(p[1] for p in all_points) or 1
        span = max_step - min_step or 1
        lines, legend = [], []
        for idx, (name, points) in enumerate(series):
            color = palette[idx % len(palette)]
            coords = [(55 + (step - min_step) / span * 505, 175 - loss / max_loss * 140) for step, loss in points]
            path = ' '.join(f'{x:.2f},{y:.2f}' for x, y in coords)
            dots = ''.join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />' for x, y in coords)
            lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5" />{dots}')
            values = '、'.join(f'{step:g}: {loss:.4f}' for step, loss in points)
            legend.append(f'<li><span style="color:{color}">●</span> {html.escape(name)}：{html.escape(values)}</li>')
        unit_label = 'Number of trees' if unit == 'trees' else unit
        sample_note = f'; training sample: {sample_count} rows' if _finite(sample_count) else ''
        figures.append(f'<figure class="viz"><figcaption>Training loss: {html.escape(metric)}</figcaption><p class="muted">X-axis: {html.escape(unit_label)}; Y-axis: {html.escape(metric)}{html.escape(sample_note)}. Values may fluctuate and are not validation loss.</p><svg viewBox="0 0 590 210" role="img" aria-label="{html.escape(metric, quote=True)} Training loss curves" style="width:100%;max-width:700px"><path d="M55 35V175H560" fill="none" stroke="#aaa"/><text x="4" y="39" font-size="12">{max_loss:.4f}</text><text x="34" y="179" font-size="12">0</text><text x="55" y="196" font-size="12">{min_step:g}</text><text x="530" y="196" font-size="12">{max_step:g}</text>{"".join(lines)}</svg><ul class="loss-legend">{"".join(legend)}</ul></figure>')
    missing = f'<p class="muted">No stepwise loss recorded for: {html.escape("、".join(unavailable))}。</p>' if unavailable else ''
    if not figures:
        return '<section><h2>Training loss curves</h2><p class="muted">This run did not record stepwise loss. Supported models will show curves in new runs.</p>' + missing + '</section>'
    return '<section><h2>Training loss curves</h2>' + ''.join(figures) + missing + '</section>'


def render(root: Path, run_id: str) -> str:
    folder = classifier._folder(root, run_id)
    file = folder / 'tabular_result.json'
    if not file.is_file():
        # Preserve explicitly installed adapters' text report support.
        return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Experiment report</title><pre>' + html.escape(tabular.report(root, run_id)) + '</pre></html>'
    data = json.loads(file.read_text(encoding='utf-8'))
    plan, metrics = data['proposal'], data['metrics']
    source_path = folder / 'source.json'
    source = json.loads(source_path.read_text(encoding='utf-8')) if source_path.exists() else {}
    esc = lambda value: html.escape(str(value if value is not None else '—'), quote=True)
    def number(value):
        return f'{value:.3f}' if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else '—'
    url = source.get('url', '')
    origin = f'<a href="{esc(url)}" rel="noopener noreferrer" target="_blank">{esc(url)}</a>' if urlsplit(url).scheme in ('http', 'https') else 'User-provided data'
    task = plan.get('task', '')
    evidence = 'Local scikit-learn models were fitted and saved.' if data.get('training_type') == 'real local scikit-learn' else 'Training method: ' + str(data.get('training_type') or 'Not provided')
    if data.get('training_type') == 'mixed local model comparison':
        evidence = 'Local mixed-model comparison: scikit-learn models were fitted; pretrained models use their adapter method and may not have fine-tuned weights. The selected model passed reload verification.'
    key_metrics = [('accuracy', 'Test accuracy'), ('macro_f1', 'Test Macro F1'), ('baseline_accuracy', 'Baseline accuracy'), ('mae', 'Test MAE'), ('baseline_mae', 'Baseline MAE'), ('r2', 'Test R²'), ('silhouette', 'Silhouette score')]
    cards = ''.join(f'<div class="metric"><span>{esc(label)}</span><strong>{number(metrics[key])}</strong></div>' for key, label in key_metrics if key in metrics)
    attempts = [a for a in data.get('attempts', []) if isinstance(a.get('validation_score'), (int, float)) and math.isfinite(a['validation_score'])]
    if task == 'clustering':
        attempts = [{'model': f'{a["groups"]} clusters', 'validation_score': a['silhouette']} for a in data.get('attempts', []) if isinstance(a.get('silhouette'), (int, float)) and math.isfinite(a['silhouette']) and 'groups' in a]
    rows = ''.join(f'<tr><td>{esc(a.get("model"))}</td><td>{number(a.get("validation_score"))}</td><td>{number(a.get("validation_accuracy"))}</td><td>{number(a.get("fit_seconds"))}</td><td>{number(a.get("predict_ms_per_row"))}</td></tr>' for a in attempts)
    metric_name = 'Macro F1 (higher is better)' if task == 'classification' else 'Silhouette score (higher is better)' if task == 'clustering' else 'MAE (lower is better)'
    score_range = (0, 1) if task == 'classification' else (-1, 1) if task == 'clustering' else None
    score_chart = _chart('Model performance', [(a.get('model', ''), a.get('validation_score')) for a in attempts], fixed_range=score_range, note=metric_name)
    fit_chart = _chart('Model training time', [(a.get('model', ''), a.get('fit_seconds')) for a in attempts], unit=' seconds', note='Includes preprocessing and fitting; faster does not imply more accurate.')
    prediction_chart = _chart('Average inference time', [(a.get('model', ''), a.get('predict_ms_per_row')) for a in attempts], unit=' ms / row', note='Average per row for one validation batch, including preprocessing; not a stable latency benchmark.')
    metric_context = 'Clustering silhouette is computed on exploration data; clusters are not ground-truth classes.' if task == 'clustering' else 'Summary metrics use a held-out local test set; the baseline uses the majority class or mean.'
    comparison_context = 'Compare cluster counts using the exploration-set silhouette score.' if task == 'clustering' else f'Same validation set; select by {metric_name}. Training times are specific to this local run.'
    matrix = metrics.get('confusion_matrix')
    labels = metrics.get('labels', [])
    matrix_html = ''
    if isinstance(matrix, list) and labels and len(matrix) == len(labels) and len(labels) <= 20:
        head = ''.join(f'<th>{esc(label)}</th>' for label in labels)
        max_cell = max((cell for row in matrix if isinstance(row, list) for cell in row if _finite(cell) and cell >= 0), default=0) or 1
        def heat_cell(cell):
            if not _finite(cell) or cell < 0:
                return '<td>—</td>'
            strength = min(1, cell / max_cell)
            return f'<td class="heat-cell" style="background:rgba(104,76,174,{0.08 + .82 * strength:.3f});color:{"#fff" if strength > .65 else "#25223a"}">{esc(cell)}</td>'
        body = ''.join('<tr><th scope="row">' + esc(label) + '</th>' + ''.join(heat_cell(cell) for cell in row) + '</tr>' for label, row in zip(labels, matrix) if isinstance(row, list))
        matrix_html = f'<section><h2>Classification results</h2><p class="muted">Test-set confusion matrix: rows are actual classes, columns are predictions; darker colors indicate more samples.</p><div class="scroll"><table class="heatmap"><thead><tr><th>Actual / predicted</th>{head}</tr></thead><tbody>{body}</tbody></table></div></section>'
    split = ''.join(f'<div><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in data.get('split_counts', {}).items())
    split_chart = _split_chart(data.get('split_counts', {}))
    distribution = plan.get('target_summary', {}) if task == 'classification' else {}
    distribution_chart = _chart('Target class distribution', list(distribution.items()), unit=' rows', precision=0, note='Class counts across all valid rows, not just the test set.') if isinstance(distribution, dict) else ''
    warning_items = data.get('suggestions', [])
    existing_ablation = data.get('ablation')
    if isinstance(existing_ablation, dict) and existing_ablation.get('selected_profile') == 'reduced':
        removed = existing_ablation.get('removed_features', [])
        warning_items = [next((f'Original plan: {warning} Final plan excluded {name}。' for name in removed if warning.startswith(name + ' ')), warning) for warning in warning_items]
    warnings = ''.join(f'<li>{esc(_english_note(w))}</li>' for w in warning_items) or '<li>No additional notes.</li>'
    backend_notes = ''.join(f'<li>{esc(a.get("model"))}：{esc(a.get("model_source"))}; method: {esc(a.get("method"))}; weights fine-tuned: {"yes" if a.get("weight_finetuning") else "no"}; context: {esc(a.get("context_rows"))} rows、{esc(a.get("context_features"))} transformed features.</li>' for a in attempts if a.get('backend'))
    comparison = (f'{score_chart}<div class="viz-grid">{fit_chart}{prediction_chart}</div><div class="scroll"><table><thead><tr><th>Model</th><th>{metric_name}</th><th>Validation accuracy</th><th>Training seconds</th><th>Inference ms / row</th></tr></thead><tbody>{rows}</tbody></table></div>' + (f'<ul>{backend_notes}</ul>' if backend_notes else '') if attempts else '<p class="muted">No comparable validation scores were recorded.</p>')
    ablation_html = ''
    ablation = data.get('ablation')
    if isinstance(ablation, dict) and ablation.get('status') == 'completed':
        reduced = {a.get('model'): a for a in ablation.get('reduced', [])}
        ablation_rows = ''.join(f'<tr><td>{esc(before.get("model"))}</td><td>{number(before.get("validation_score"))}</td><td>{number(reduced.get(before.get("model"), {}).get("validation_score"))}</td><td>{number(before.get("fit_seconds"))}</td><td>{number(reduced.get(before.get("model"), {}).get("fit_seconds"))}</td></tr>' for before in ablation.get('baseline', []))
        before = ablation.get('baseline', [])
        after = ablation.get('reduced', [])
        score_before = [{'model': x.get('model'), 'value': x.get('validation_score')} for x in before]
        score_after = [{'model': x.get('model'), 'value': x.get('validation_score')} for x in after]
        time_before = [{'model': x.get('model'), 'value': x.get('fit_seconds')} for x in before]
        time_after = [{'model': x.get('model'), 'value': x.get('fit_seconds')} for x in after]
        ablation_score = _paired_chart('Ablation: validation performance', score_before, score_after, fixed_range=score_range, note=metric_name + '. Purple: before; teal: after.')
        ablation_time = _paired_chart('Ablation: training time', time_before, time_after, unit=' seconds', note='Purple: before; teal: after. Each was run once; system load can affect time.')
        label = 'After' if ablation.get('selected_profile') == 'reduced' else 'Before'
        ablation_note = 'Both profiles use the same training and validation split. Only the selected profile and fixed baseline are evaluated on the test set. Repeated validation can cause selection bias.'
        ablation_html = f'<section><h2>Feature ablation comparison</h2><p>Excluded after adjustment: {esc("、".join(ablation.get("removed_features", [])))}; selected on validation set: <strong>{label}</strong>。</p>{ablation_score}{ablation_time}<div class="scroll"><table><thead><tr><th>Model</th><th>Before {metric_name}</th><th>After {metric_name}</th><th>Before seconds</th><th>After seconds</th></tr></thead><tbody>{ablation_rows}</tbody></table></div><p class="muted">{esc(ablation_note)}</p></section>'
    discovery_html = ''
    discovery_path = folder / 'model_discovery.json'
    if discovery_path.is_file():
        discovery = json.loads(discovery_path.read_text(encoding='utf-8'))
        found = ''
        for item in discovery.get('results', [])[:5]:
            link = item.get('url', '')
            if urlsplit(link).scheme == 'https' and urlsplit(link).hostname == 'huggingface.co':
                found += f'<li><a href="{esc(link)}" rel="noopener noreferrer" target="_blank">{esc(item.get("id"))}</a> · License: {esc(item.get("license"))}</li>'
        state = 'Searched public model information (' + esc(discovery.get('source') or 'source unspecified') + '）。' if discovery.get('status') == 'completed' else 'Model search did not complete.'
        discovery_html = f'<section><h2>Hugging Face model search</h2><p>{state} {esc(discovery.get("decision"))}</p><ul>{found}</ul></section>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Training experiment report</title><style>
    :root{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#25223a;background:#f6f5fa}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1000px;margin:auto;padding:38px 24px 80px}}header{{border-bottom:1px solid #ddd9e8;padding-bottom:24px}}.eyebrow{{color:#7564ad;font-weight:700;font-size:13px;letter-spacing:.06em}}h1{{font-size:clamp(28px,4vw,42px);margin:8px 0}}h2{{font-size:20px;margin:0 0 16px}}p{{line-height:1.6}}.muted{{color:#716d80;font-size:13px}}section{{background:#fff;border:1px solid #e8e5ef;border-radius:18px;padding:24px;margin-top:18px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}}.metric{{background:#f6f3ff;border-radius:12px;padding:15px}}.metric span,.metric strong{{display:block}}.metric span{{color:#706881;font-size:13px}}.metric strong{{font-size:28px;margin-top:4px;color:#49347e}}.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}.meta div,.split div{{padding:10px 0;border-bottom:1px solid #efedf4}}.meta span,.split span{{display:block;color:#777285;font-size:12px}}.meta b,.split b{{display:block;margin-top:5px;overflow-wrap:anywhere}}.split{{display:flex;gap:24px;flex-wrap:wrap}}.bar-row{{display:grid;grid-template-columns:160px 1fr 55px;gap:12px;align-items:center;margin:10px 0;font-size:13px}}.track{{height:9px;background:#eeeaf6;border-radius:20px;overflow:hidden}}.bar{{height:100%;background:#8068c2;border-radius:20px}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;text-align:left}}th,td{{border-bottom:1px solid #efedf4;padding:10px;white-space:nowrap}}th{{color:#706881}}li{{line-height:1.6;margin:7px 0}}a{{color:#6344ab;overflow-wrap:anywhere}}@media(max-width:560px){{main{{padding:26px 15px 60px}}section{{padding:18px}}.bar-row{{grid-template-columns:110px 1fr 43px}}}}@media print{{body{{background:white}}section{{break-inside:avoid}}}}
    .viz{{margin:16px 0;padding:16px;background:#faf9fd;border:1px solid #efedf4;border-radius:12px}}.viz figcaption{{font-weight:700}}.viz p{{margin:6px 0 12px}}.viz-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:14px}}.viz-row{{display:grid;grid-template-columns:minmax(90px,150px) minmax(60px,1fr) auto;gap:10px;align-items:center;margin:10px 0;font-size:13px}}.viz-label{{overflow-wrap:anywhere}}.viz-track{{position:relative;height:10px;background:#eeeaf6;border-radius:8px}}.viz-zero,.viz-bar{{position:absolute;display:block;height:100%}}.viz-zero{{width:1px;background:#81788d}}.viz-bar{{background:#8068c2;border-radius:8px}}.viz.paired .viz-row:nth-of-type(even) .viz-bar{{background:#3f9992}}.split-bar{{display:flex;height:14px;overflow:hidden;border-radius:8px;margin:14px 0}}.split-segment{{display:block;height:100%}}.split-0{{background:#8068c2}}.split-1{{background:#55a6a0}}.split-2{{background:#e8a56a}}.split-3{{background:#b5a2d2}}.split-legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:13px}}.split-dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}.heatmap{{width:auto}}.heatmap td{{text-align:center;border:3px solid white;border-radius:6px;font-weight:700;min-width:58px}}@media(max-width:560px){{.viz-row{{grid-template-columns:minmax(70px,110px) minmax(40px,1fr) auto;font-size:12px}}}}
    </style></head><body><main><header><div class="eyebrow">TRAINPILOT · EXPERIMENT</div><h1>Training experiment report</h1><p class="muted">Experiment {esc(run_id)} · Results are from this local run</p><p>{esc(plan.get('goal') or 'Goal not provided')}</p></header>
    <section><h2>Results summary</h2><p>Selected model: <strong>{esc(plan.get('model'))}</strong> · Task: {esc(task)} · Target: {esc(plan.get('target') or 'None')}</p><p class="muted">{esc(evidence)}</p><div class="metrics">{cards or '<p class="muted">No summary metrics were recorded.</p>'}</div><p class="muted">{metric_context}</p></section>
    <section><h2>Data and split</h2><div class="meta"><div><span>Data source</span><b>{origin}</b></div><div><span>Valid rows</span><b>{esc(plan.get('rows'))} rows</b></div><div><span>Input features</span><b>{esc('、'.join(plan.get('features', [])))}</b></div></div><div class="split">{split}</div>{split_chart}{distribution_chart}</section>
    <section><h2>Model comparison</h2><p class="muted">{comparison_context}</p>{comparison}</section>{_loss_charts(attempts)}{ablation_html}{discovery_html}{matrix_html}
    <section><h2>Evaluation notes</h2><ul>{warnings}</ul><p class="muted">Model reload verification: {'Passed' if data.get('checkpoint_reloaded') else 'Not passed'}. Evaluate generalization on new independent data.</p></section></main></body></html>'''
