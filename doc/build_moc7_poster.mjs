#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
let artifactToolPath;
try {
  artifactToolPath = require.resolve("@oai/artifact-tool");
} catch (error) {
  throw new Error(
    "@oai/artifact-tool is required. Set NODE_PATH or " +
      "ARTIFACT_TOOL_NODE_MODULES to its node_modules directory.",
    { cause: error },
  );
}

const { SpreadsheetFile, Workbook } = await import(
  pathToFileURL(artifactToolPath).href
);

const [sourceArg = "doc/moc7_poster.md", outputArg = "doc/moc7_poster.xlsx"] =
  process.argv.slice(2);
const sourcePath = path.resolve(sourceArg);
const outputPath = path.resolve(outputArg);
const markdown = await fs.readFile(sourcePath, "utf8");

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function extractSection(title, level) {
  const marker = "#".repeat(level);
  const startPattern = new RegExp(`^${marker} ${escapeRegex(title)}\\s*$`, "m");
  const match = startPattern.exec(markdown);
  if (!match) throw new Error(`Missing Markdown section: ${title}`);
  const bodyStart = match.index + match[0].length;
  const rest = markdown.slice(bodyStart);
  const endPattern = new RegExp(`^#{1,${level}} \\S`, "m");
  const end = endPattern.exec(rest);
  return rest.slice(0, end ? end.index : undefined).trim();
}

function plainInline(value) {
  return value
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .trim();
}

function prose(value) {
  return value
    .split(/\n\s*\n/)
    .map((paragraph) => plainInline(paragraph.replace(/\s*\n\s*/g, " ")))
    .filter(Boolean)
    .join("\n\n");
}

function parseTable(section) {
  const rows = section
    .split("\n")
    .filter((line) => /^\s*\|.*\|\s*$/.test(line))
    .map((line) =>
      line
        .trim()
        .slice(1, -1)
        .split("|")
        .map((cell) => cell.trim()),
    );
  if (rows.length < 2) throw new Error("Expected a Markdown table");
  return rows.filter(
    (row, index) =>
      index === 0 || !row.every((cell) => /^:?-{3,}:?$/.test(cell)),
  );
}

function firstLink(value) {
  const match = /\[([^\]]+)\]\(([^)]+)\)/.exec(value);
  return match ? { label: plainInline(match[1]), url: match[2] } : null;
}

const coverEmail = prose(extractSection("Cover Email", 2));
const applicantRows = parseTable(extractSection("Applicant Details", 2));
const presentationTitle = prose(extractSection("Presentation Title", 3));
const keywords = prose(extractSection("Keywords", 3));
const summary = prose(extractSection("Summary", 3));
const abstract = prose(extractSection("Abstract", 3));
const supportingRows = parseTable(
  extractSection("Papers, Preprints, and Supporting Work", 3),
);
const submissionRows = parseTable(extractSection("Submission Check", 2));

if (applicantRows.length < 10 || supportingRows.length < 3) {
  throw new Error("The application source appears incomplete");
}

const workbook = Workbook.create();
const applicationSheet = workbook.worksheets.add("Application");
const coverSheet = workbook.worksheets.add("Cover Email");
const summarySheet = workbook.worksheets.add("Summary");
const abstractSheet = workbook.worksheets.add("Abstract");
const supportingSheet = workbook.worksheets.add("Supporting Work");
const checkSheet = workbook.worksheets.add("Submission Check");

const palette = {
  navy: "#17324D",
  teal: "#287C7D",
  gold: "#D9A441",
  pale: "#EEF5F5",
  paleGold: "#FBF4E4",
  ink: "#1F2937",
  muted: "#667085",
  rule: "#CBD5E1",
  white: "#FFFFFF",
};

const titleFont = { name: "Aptos Display", size: 22, bold: true, color: palette.white };
const sectionFont = { name: "Aptos", size: 12, bold: true, color: palette.white };
const bodyFont = { name: "Aptos", size: 10, color: palette.ink };

function titleBand(sheet, subtitle) {
  sheet.showGridLines = false;
  sheet.getRange("A1:F2").merge();
  sheet.getRange("A1").values = [["Models of Consciousness 7 — Poster Application"]];
  sheet.getRange("A1:F2").format = {
    fill: palette.navy,
    font: titleFont,
    verticalAlignment: "center",
    horizontalAlignment: "left",
  };
  sheet.getRange("A3:F3").merge();
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange("A3:F3").format = {
    fill: palette.pale,
    font: { ...bodyFont, italic: true, color: palette.teal },
    verticalAlignment: "center",
  };
  sheet.getRange("A1:F3").format.borders = {
    bottom: { style: "medium", color: palette.gold },
  };
  sheet.getRange("A1:F2").format.rowHeight = 28;
  sheet.getRange("A3:F3").format.rowHeight = 24;
}

function sectionBand(sheet, range, label) {
  range.merge();
  range.values = [[label]];
  range.format = {
    fill: palette.teal,
    font: sectionFont,
    verticalAlignment: "center",
  };
  range.format.rowHeight = 22;
}

function styleTable(range, headerRange) {
  range.format = {
    font: bodyFont,
    verticalAlignment: "top",
    wrapText: true,
    borders: {
      outside: { style: "thin", color: palette.rule },
      insideHorizontal: { style: "thin", color: palette.rule },
    },
  };
  headerRange.format = {
    fill: palette.navy,
    font: { ...bodyFont, bold: true, color: palette.white },
    verticalAlignment: "center",
  };
}

titleBand(applicationSheet, "Submission-ready fields generated from doc/moc7_poster.md");
sectionBand(applicationSheet, applicationSheet.getRange("A5:F5"), "Applicant details");
const applicantData = applicantRows.map((row) => row.map(plainInline));
applicationSheet
  .getRangeByIndexes(5, 0, applicantData.length, 2)
  .writeValues(applicantData);
styleTable(
  applicationSheet.getRangeByIndexes(5, 0, applicantData.length, 2),
  applicationSheet.getRange("A6:B6"),
);
applicationSheet.getRange("A6:A20").format.fill = palette.pale;
applicationSheet.getRange("A6:A20").format.font = { ...bodyFont, bold: true };
const bookletRow = 6 + applicantData.length + 1;
sectionBand(
  applicationSheet,
  applicationSheet.getRangeByIndexes(bookletRow, 0, 1, 6),
  "Conference booklet",
);
applicationSheet.getRangeByIndexes(bookletRow + 1, 0, 2, 2).values = [
  ["Presentation title", presentationTitle],
  ["Keywords", keywords],
];
applicationSheet.getRangeByIndexes(bookletRow + 1, 0, 2, 2).format = {
  font: bodyFont,
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: palette.rule },
};
applicationSheet.getRangeByIndexes(bookletRow + 1, 0, 2, 1).format = {
  fill: palette.paleGold,
  font: { ...bodyFont, bold: true },
};
applicationSheet.getRange("A:A").format.columnWidth = 34;
applicationSheet.getRange("B:B").format.columnWidth = 78;
applicationSheet.getRange("C:F").format.columnWidth = 3;
applicationSheet.getRange(`A6:B${bookletRow + 3}`).format.autofitRows();
applicationSheet.freezePanes.freezeRows(5);

titleBand(coverSheet, "Optional late-submission email");
sectionBand(coverSheet, coverSheet.getRange("A5:F5"), "Cover email");
coverSheet.getRange("A6:F20").merge();
coverSheet.getRange("A6").values = [[coverEmail]];
coverSheet.getRange("A6:F20").format = {
  fill: palette.white,
  font: { ...bodyFont, size: 11 },
  verticalAlignment: "top",
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: palette.rule },
};
coverSheet.getRange("A:F").format.columnWidth = 18;
coverSheet.getRange("A6:F20").format.rowHeight = 22;

titleBand(summarySheet, "Booklet summary and word-limit control");
sectionBand(summarySheet, summarySheet.getRange("A5:F5"), presentationTitle);
summarySheet.getRange("A6:F15").merge();
summarySheet.getRange("A6").values = [[summary]];
summarySheet.getRange("A6:F15").format = {
  fill: palette.white,
  font: { ...bodyFont, size: 11 },
  verticalAlignment: "top",
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: palette.rule },
};
summarySheet.getRange("A17:B19").values = [
  ["Word limit", 250],
  ["Summary words", null],
  ["Status", null],
];
summarySheet.getRange("B18").formulas = [
  ['=IF(TRIM(A6)="",0,LEN(TRIM(A6))-LEN(SUBSTITUTE(TRIM(A6)," ",""))+1)'],
];
summarySheet.getRange("B19").formulas = [
  ['=IF(B18<=B17,"Complete — within limit","Review — over limit")'],
];
summarySheet.getRange("A17:B19").format = {
  font: bodyFont,
  borders: { preset: "outside", style: "thin", color: palette.rule },
};
summarySheet.getRange("A17:A19").format = {
  fill: palette.paleGold,
  font: { ...bodyFont, bold: true },
};
summarySheet.getRange("B17:B18").format.numberFormat = "0";
summarySheet.getRange("A:F").format.columnWidth = 18;
summarySheet.getRange("A6:F15").format.rowHeight = 24;

titleBand(abstractSheet, "Booklet abstract and word-limit control");
sectionBand(abstractSheet, abstractSheet.getRange("A5:F5"), presentationTitle);
abstractSheet.getRange("A6:F17").merge();
abstractSheet.getRange("A6").values = [[abstract]];
abstractSheet.getRange("A6:F17").format = {
  fill: palette.white,
  font: { ...bodyFont, size: 11 },
  verticalAlignment: "top",
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: palette.rule },
};
abstractSheet.getRange("A19:B21").values = [
  ["Word limit", 250],
  ["Abstract words", null],
  ["Status", null],
];
abstractSheet.getRange("B20").formulas = [
  ['=IF(TRIM(A6)="",0,LEN(TRIM(A6))-LEN(SUBSTITUTE(TRIM(A6)," ",""))+1)'],
];
abstractSheet.getRange("B21").formulas = [
  ['=IF(B20<=B19,"Complete — within limit","Review — over limit")'],
];
abstractSheet.getRange("A19:B21").format = {
  font: bodyFont,
  borders: { preset: "outside", style: "thin", color: palette.rule },
};
abstractSheet.getRange("A19:A21").format = {
  fill: palette.paleGold,
  font: { ...bodyFont, bold: true },
};
abstractSheet.getRange("B19:B20").format.numberFormat = "0";
abstractSheet.getRange("A:F").format.columnWidth = 18;
abstractSheet.getRange("A6:F17").format.rowHeight = 24;

titleBand(supportingSheet, "Public foundations and implementation");
sectionBand(supportingSheet, supportingSheet.getRange("A5:F5"), "Supporting work");
const supportingData = [
  ["Work", "Relevance", "Source URL"],
  ...supportingRows.slice(1).map(([work, relevance]) => {
    const link = firstLink(work);
    return [plainInline(work), plainInline(relevance), link?.url ?? ""];
  }),
];
supportingSheet
  .getRangeByIndexes(5, 0, supportingData.length, 3)
  .writeValues(supportingData);
styleTable(
  supportingSheet.getRangeByIndexes(5, 0, supportingData.length, 3),
  supportingSheet.getRange("A6:C6"),
);
supportingSheet.getRange("A:A").format.columnWidth = 42;
supportingSheet.getRange("B:B").format.columnWidth = 72;
supportingSheet.getRange("C:C").format.columnWidth = 62;
supportingSheet.getRange("D:F").format.columnWidth = 3;
supportingSheet.getRange(`A6:C${5 + supportingData.length}`).format.autofitRows();
supportingSheet.freezePanes.freezeRows(6);

titleBand(checkSheet, "Completion and follow-up checklist");
sectionBand(checkSheet, checkSheet.getRange("A5:F5"), "Submission check");
const submissionData = submissionRows.map((row) => row.map(plainInline));
checkSheet
  .getRangeByIndexes(5, 0, submissionData.length, 2)
  .writeValues(submissionData);
styleTable(
  checkSheet.getRangeByIndexes(5, 0, submissionData.length, 2),
  checkSheet.getRange("A6:B6"),
);
const abstractCheckIndex = submissionData.findIndex(
  ([requirement]) => requirement === "Abstract within 250-word limit",
);
const summaryCheckIndex = submissionData.findIndex(
  ([requirement]) => requirement === "Summary within 250-word limit",
);
if (summaryCheckIndex >= 1) {
  checkSheet.getRangeByIndexes(5 + summaryCheckIndex, 1, 1, 1).formulas = [
    ["='Summary'!B19"],
  ];
}
if (abstractCheckIndex >= 1) {
  checkSheet.getRangeByIndexes(5 + abstractCheckIndex, 1, 1, 1).formulas = [
    ["='Abstract'!B21"],
  ];
}
checkSheet.getRange("A:A").format.columnWidth = 48;
checkSheet.getRange("B:B").format.columnWidth = 56;
checkSheet.getRange("C:F").format.columnWidth = 3;
checkSheet.getRange(`A6:B${5 + submissionData.length}`).format.autofitRows();
checkSheet.freezePanes.freezeRows(6);

const inspect = await workbook.inspect({
  kind: "table",
  range: "Summary!A17:B19",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 4,
  maxChars: 3000,
});
const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
  maxChars: 3000,
});
if (/"matchCount"\s*:\s*[1-9]/.test(errorScan.ndjson)) {
  throw new Error(`Formula errors detected:\n${errorScan.ndjson}`);
}

const previewDir = process.env.MOC7_PREVIEW_DIR;
if (previewDir) {
  await fs.mkdir(previewDir, { recursive: true });
  for (const sheetName of [
    "Application",
    "Cover Email",
    "Summary",
    "Abstract",
    "Supporting Work",
    "Submission Check",
  ]) {
    const preview = await workbook.render({
      sheetName,
      autoCrop: "all",
      scale: 1,
      format: "png",
    });
    await fs.writeFile(
      path.join(previewDir, `${sheetName.toLowerCase().replaceAll(" ", "-")}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
// Some artifact-tool builds spill a verbose inspection transcript beside the
// requested output. It is QA scratch data, not a documentation artifact.
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });

process.stdout.write(`Built ${path.relative(process.cwd(), outputPath)}\n`);
process.stdout.write(`${inspect.ndjson}\n`);
