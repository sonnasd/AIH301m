import fs from 'node:fs/promises';
import { ROOT, BUILD, SKILL, RUNTIME, finalizePresentation } from './layout.mjs';

const req = JSON.parse(await fs.readFile(`${BUILD}/requirements.json`, 'utf8'));
process.env.RUNTIME_NODE_MODULES = `${RUNTIME}/node/node_modules`;
const finalPath = `${ROOT}/output/presentations/AI_Clinician_Nhom_3.pptx`;
await fs.mkdir(`${ROOT}/output/presentations`, { recursive: true });
const result = await finalizePresentation({
  workspaceDir: ROOT,
  candidatePath: `${BUILD}/candidate.pptx`,
  finalPath,
  pythonExecutable: `${RUNTIME}/python/python.exe`,
  integrityValidatorPath: `${SKILL}/container_tools/inspect_presentation_package_integrity.py`,
  layoutValidatorPath: `${SKILL}/container_tools/inspect_presentation_layout_geometry.py`,
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit', ...req.tableOwners.flatMap(n => ['--require-native-table-slide', String(n)])],
  explicitTotalSlideCount: 32,
  requiredNativeTableOwnerSlides: req.tableOwners,
  requiredNativeChartOwnerSlides: req.chartOwners,
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: 'user_request', families: ['Arial'] },
  verifyArtifactToolImport: true,
  receiptPath: `${BUILD}/validation.json`,
});
console.log(JSON.stringify(result, null, 2));
