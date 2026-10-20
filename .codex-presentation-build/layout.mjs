import { pathToFileURL } from 'node:url';

export const ROOT = 'C:/Users/ADMIN/Desktop/ass2';
export const BUILD = `${ROOT}/.codex-presentation-build`;
export const SKILL = 'C:/Users/ADMIN/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
export const RUNTIME = 'C:/Users/ADMIN/.cache/codex-runtimes/codex-primary-runtime/dependencies';
export const { Presentation, PresentationFile } = await import(pathToFileURL(`${RUNTIME}/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs`));
export const { applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(`${SKILL}/container_tools/artifact_tool_utils.mjs`));
export const C = { ink: '#263238', teal: '#087F83', red: '#8C2835', muted: '#58656A', grid: '#D5DFE0', pale: '#EEF6F6', white: '#FFFFFF' };
export const tableOwners = [];
export const chartOwners = [];
export const metadata = [];

export function text(s, value, x, y, w, h, pt = 22, color = C.ink, bold = false) {
  const shape = s.shapes.add({ geometry: 'textbox', name: `text-${s.shapes.items.length + 1}`, position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  shape.text = value;
  shape.text.style = { typeface: 'Arial', fontFamily: 'Arial', fontSize: pt * 4 / 3, color, bold, autoFit: 'none', wrap: 'square', verticalAlignment: 'top', insets: 0 };
  return shape;
}

export function slide(p, number, title, seconds, notes, sources, speaker) {
  sources = sources.replaceAll(' • ', ', ');
  const s = p.slides.add();
  s.background.fill = C.white;
  text(s, title, 64, 44, 1152, 90, 34, C.ink, true);
  const who = speaker ?? (number <= 9 ? 'Bùi Đức Giang' : number <= 20 ? 'Nguyễn Huy Sơn' : number <= 28 ? 'Cường' : 'Nhóm');
  text(s, sources, 64, 660, 1050, 42, 12, C.muted);
  text(s, String(number).padStart(2, '0'), 1160, 662, 56, 24, 12, C.muted);
  s.speakerNotes.text = `Người trình bày: ${who}\nThời lượng dự kiến: ${seconds} giây${number > 28 ? ' (phụ lục, ngoài 30 phút)' : ''}.\nÝ chính để tự nói:\n${notes}\nNguồn: ${sources}\nThư mục gốc của các đường dẫn: C:\\Users\\ADMIN\\Desktop\\ass2.\nBài báo gốc: docs/AI_in_Healthcare_Paper for Assignment 2.pdf. DOI: 10.1038/s41591-018-0213-5.`;
  metadata.push({ number, title, seconds, speaker: who, sources });
  return s;
}

export function rows(s, items, { x = 64, y = 162, w = 1140, gap = 108, pt = 22 } = {}) {
  items.forEach((item, i) => {
    if (Array.isArray(item)) {
      text(s, item[0], x, y + i * gap, w, 36, pt, C.teal, true);
      text(s, item[1], x, y + i * gap + 42, w, gap - 42, pt);
    } else text(s, item, x, y + i * gap, w, gap - 10, pt);
  });
}

export function table(s, number, values, { x = 64, y = 176, w = 1152, h = 330, widths, pt = 21 } = {}) {
  const t = s.tables.add({ rows: values.length, columns: values[0].length, left: x, top: y, width: w, height: h, values, columnWidths: widths });
  t.borders.assign({ fill: C.grid, width: 0.7, style: 'solid' });
  for (let r = 0; r < values.length; r++) {
    t.rows[r].height = h / values.length;
    for (let c = 0; c < values[r].length; c++) {
      const cell = t.getCell(r, c);
      cell.fill = r === 0 ? C.pale : C.white;
      cell.text.style = { typeface: 'Arial', fontFamily: 'Arial', fontSize: pt * 4 / 3, color: r === 0 ? C.teal : C.ink, bold: r === 0, insets: 10, autoFit: 'none' };
    }
  }
  tableOwners.push(number);
  return t;
}

export function chart(s, number, categories, series, { x = 64, y = 170, w = 730, h = 380, min = 0, max, format = '0', title = '', legend = false } = {}) {
  const ch = s.charts.add('bar', {
    position: { left: x, top: y, width: w, height: h }, categories, series,
    title: title || undefined, titleTextStyle: { fontSize: 26.67, fill: C.ink },
    hasLegend: legend, legend: { position: 'bottom', textStyle: { fontSize: 26.67, fill: C.ink } },
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 100 },
    xAxis: { textStyle: { fontSize: 26.67, fill: C.ink }, line: { fill: C.grid, width: 1 }, majorGridlines: null },
    yAxis: { min, max, numberFormatCode: format, textStyle: { fontSize: 26.67, fill: C.muted }, majorGridlines: { fill: C.grid, width: 0.8 } },
    dataLabels: { showValue: true, position: 'outEnd', numberFormatCode: format, textStyle: { fontSize: 26.67, fill: C.ink, bold: true } },
    chartFill: C.white, chartLine: { fill: 'none', width: 0 }, plotAreaFill: C.white, plotAreaLine: { fill: 'none', width: 0 },
  });
  applyPresentationChartFont(ch, { fontFamily: 'Arial' });
  chartOwners.push(number);
  return ch;
}

export function equation(s, formula, meaning, y = 190, pt = 26) {
  text(s, formula, 64, y, 1150, 90, pt, C.teal, true);
  text(s, meaning, 64, y + 104, 1130, 128, 22);
}

export function node(s, label, x, y, w = 330, h = 100) {
  s.shapes.add({ geometry: 'rect', position: { left: x, top: y, width: w, height: h }, fill: C.white, line: { fill: C.teal, width: 1.5 } });
  text(s, label, x + 18, y + 18, w - 36, h - 24, 22, C.ink, true);
}

export function connector(s, x1, y1, x2, y2) {
  s.shapes.add({ geometry: 'line', position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 }, line: { fill: C.teal, width: 2 } });
}
